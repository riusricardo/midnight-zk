//! GPU evaluation of the quotient numerator.
//!
//! [`try_compute_h_poly`] runs the prover's whole quotient-numerator evaluation
//! (custom gates, permutation, lookups, trash, for every circuit) on the GPU:
//! one straight-line program per proof, run on every row of the extended domain
//! by the CUDA backend's graph-evaluation kernel (design: the backend's
//! `docs/gpu-integration/GPU_QUOTIENT_EVALUATION.md`). The columns stay on the
//! device: the proving key's extended-domain columns (fixed cosets, permutation
//! cosets, Lagrange selectors) are uploaded once and reused by every proof with
//! the key ([`KeyColumns`], within the backend's process-wide budget), and the
//! proof's own columns (advice, instance, permutation products, lookup and
//! trash polynomials) are moved into the extended coset on the device from
//! their coefficients.
//!
//! Field arithmetic is exact, so each row's value equals the CPU path's bit for
//! bit even where the program groups the operations differently. It runs only
//! when the field is midnight's BLS12-381 scalar field, the GPU path is chosen
//! for this size (`should_use_gpu_ntt`: never in CPU mode, so CPU mode does not
//! touch the GPU), the installed backend has the kernel and the device has the
//! memory. On a GPU failure it reports the fallback (fatal under
//! `MIDNIGHT_GPU_STRICT`) and returns `None`, and the caller runs the CPU path
//! from the start.

use std::sync::{Arc, Mutex};

use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use midnight_bls12_381_cuda::config::note_gpu_fallback;

use super::{
    evaluation::{Calculation, Evaluator, GraphEvaluator, ValueSource},
    lookup, permutation, trash, Any, ConstraintSystem, Expression, ProvingKey,
};
use crate::{
    gpu_accel::{
        backend::{
            curves::Fq,
            graph_eval::{
                self as ge, op, src, Calc, CacheReservation, Column, DeviceColumns, GraphProgram, Limbs,
                MixedColumns, Src,
            },
        },
        should_use_gpu_ntt, try_as_fq_slice, try_as_fq_slice_mut,
    },
    poly::{commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, Polynomial},
};

/// Device memory kept free beyond the estimate of a proof's needs.
const MEMORY_MARGIN: u64 = 128 << 20;

/// Builds a straight-line program in the kernel's encoding.
#[derive(Default)]
struct Builder {
    calcs: Vec<Calc>,
    parts: Vec<Src>,
    constants: Vec<Limbs>,
    rotations: Vec<i32>,
    num_intermediates: u32,
}

impl Builder {
    fn constant(&mut self, c: Limbs) -> Src {
        let i = match self.constants.iter().position(|x| *x == c) {
            Some(i) => i,
            None => {
                self.constants.push(c);
                self.constants.len() - 1
            }
        };
        Src::indexed(src::CONSTANT, i)
    }

    fn rotation(&mut self, r: i32) -> usize {
        match self.rotations.iter().position(|x| *x == r) {
            Some(i) => i,
            None => {
                self.rotations.push(r);
                self.rotations.len() - 1
            }
        }
    }

    fn column(&mut self, kind: u32, index: usize, rotation: i32) -> Src {
        let r = self.rotation(rotation);
        Src::column(kind, index, r)
    }

    fn fresh(&mut self) -> u32 {
        self.num_intermediates += 1;
        self.num_intermediates - 1
    }

    fn emit(&mut self, o: u32, a: Src, b: Src) -> Src {
        let target = self.fresh();
        self.calcs.push(Calc { op: o, target, a, b, ..Default::default() });
        Src::indexed(src::INTERMEDIATE, target as usize)
    }

    fn add(&mut self, a: Src, b: Src) -> Src {
        self.emit(op::ADD, a, b)
    }
    fn sub(&mut self, a: Src, b: Src) -> Src {
        self.emit(op::SUB, a, b)
    }
    fn mul(&mut self, a: Src, b: Src) -> Src {
        self.emit(op::MUL, a, b)
    }
    fn square(&mut self, a: Src) -> Src {
        self.emit(op::SQUARE, a, Src::default())
    }

    /// `value * y + term`: one constraint folded into the running value.
    fn fold(&mut self, value: Src, term: Src) -> Src {
        let scaled = self.mul(value, Src::scalar(src::Y));
        self.add(scaled, term)
    }

    /// Inlines a `GraphEvaluator` (its constants, rotations and intermediates
    /// remapped) with `previous` for its previous value; returns its result
    /// (zero for an empty evaluator, as `GraphEvaluator::evaluate` returns).
    fn inline(&mut self, ev: &GraphEvaluator<Fq>, previous: Src) -> Src {
        let base = self.num_intermediates;
        self.num_intermediates += ev.num_intermediates as u32;
        let constants: Vec<Src> = ev.constants.iter().map(|c| self.constant(ge::fq_limbs(c))).collect();
        let rotations: Vec<usize> = ev.rotations.iter().map(|r| self.rotation(*r)).collect();
        let map = |v: &ValueSource| -> Src {
            match *v {
                ValueSource::Constant(i) => constants[i],
                ValueSource::Intermediate(i) => Src::indexed(src::INTERMEDIATE, base as usize + i),
                ValueSource::Fixed(c, r) => Src::column(src::FIXED, c, rotations[r]),
                ValueSource::Advice(c, r) => Src::column(src::ADVICE, c, rotations[r]),
                ValueSource::Instance(c, r) => Src::column(src::INSTANCE, c, rotations[r]),
                ValueSource::Challenge(i) => Src::indexed(src::CHALLENGE, i),
                ValueSource::Beta() => Src::scalar(src::BETA),
                ValueSource::Gamma() => Src::scalar(src::GAMMA),
                ValueSource::Theta() => Src::scalar(src::THETA),
                ValueSource::TrashChallenge() => Src::scalar(src::TRASH_CHALLENGE),
                ValueSource::Y() => Src::scalar(src::Y),
                ValueSource::PreviousValue() => previous,
            }
        };
        for info in &ev.calculations {
            let mut c = Calc { target: base + info.target as u32, ..Default::default() };
            match &info.calculation {
                Calculation::Add(a, b) => (c.op, c.a, c.b) = (op::ADD, map(a), map(b)),
                Calculation::Sub(a, b) => (c.op, c.a, c.b) = (op::SUB, map(a), map(b)),
                Calculation::Mul(a, b) => (c.op, c.a, c.b) = (op::MUL, map(a), map(b)),
                Calculation::Square(a) => (c.op, c.a) = (op::SQUARE, map(a)),
                Calculation::Double(a) => (c.op, c.a) = (op::DOUBLE, map(a)),
                Calculation::Negate(a) => (c.op, c.a) = (op::NEGATE, map(a)),
                Calculation::Store(a) => (c.op, c.a) = (op::STORE, map(a)),
                Calculation::Horner(start, ps, factor) => {
                    (c.op, c.a, c.c) = (op::HORNER, map(start), map(factor));
                    c.parts_offset = self.parts.len() as u32;
                    c.parts_len = ps.len() as u32;
                    let parts: Vec<Src> = ps.iter().map(map).collect();
                    self.parts.extend(parts);
                }
            }
            self.calcs.push(c);
        }
        match ev.calculations.last() {
            Some(last) => Src::indexed(src::INTERMEDIATE, (base + last.target as u32) as usize),
            None => self.constant(ge::fq_limbs(&Fq::ZERO)),
        }
    }

    /// The program, whose last calculation writes `result` (the row's value),
    /// with its intermediates packed into reused slots.
    fn finish(mut self, result: Src) -> GraphProgram {
        let last_writes_result = self.calcs.last().is_some_and(|c| {
            result.kind == src::INTERMEDIATE && c.target == result.index
        });
        if !last_writes_result {
            let target = self.fresh();
            self.calcs.push(Calc { op: op::STORE, target, a: result, ..Default::default() });
        }
        let slots = allocate_slots(&mut self.calcs, &mut self.parts, self.num_intermediates);
        GraphProgram {
            calcs: self.calcs,
            parts: self.parts,
            constants: self.constants,
            rotations: self.rotations,
            num_intermediates: slots.max(1),
        }
    }
}

/// Calls `f` on every source calculation `c` reads.
fn for_each_read(c: &mut Calc, parts: &mut [Src], mut f: impl FnMut(&mut Src)) {
    match c.op {
        op::ADD | op::SUB | op::MUL => {
            f(&mut c.a);
            f(&mut c.b);
        }
        op::HORNER => {
            f(&mut c.a);
            f(&mut c.c);
            let start = c.parts_offset as usize;
            for p in &mut parts[start..start + c.parts_len as usize] {
                f(p);
            }
        }
        _ => f(&mut c.a),
    }
}

/// Renames the intermediates so that they share slots: an intermediate's slot
/// is free again after its last read. The kernel reads all operands of a
/// calculation before it writes the target, so the target may take the slot of
/// an operand read for the last time. Fewer slots mean less scratch per row and
/// more rows in flight. Returns the number of slots. An intermediate read
/// before it is written keeps an out-of-range index, which the kernel rejects.
fn allocate_slots(calcs: &mut [Calc], parts: &mut [Src], num_intermediates: u32) -> u32 {
    const UNSET: u32 = u32::MAX;
    let n = num_intermediates as usize;
    let mut last_read = vec![usize::MAX; n];
    for (i, c) in calcs.iter_mut().enumerate() {
        for_each_read(c, parts, |s| {
            if s.kind == src::INTERMEDIATE && (s.index as usize) < n {
                last_read[s.index as usize] = i;
            }
        });
    }
    let last = calcs.len().saturating_sub(1);
    let mut slot = vec![UNSET; n];
    let mut free: Vec<u32> = Vec::new();
    let mut slots = 0u32;
    let mut released: Vec<usize> = Vec::new();
    for (i, c) in calcs.iter_mut().enumerate() {
        released.clear();
        for_each_read(c, parts, |s| {
            if s.kind == src::INTERMEDIATE && (s.index as usize) < n {
                let old = s.index as usize;
                if last_read[old] == i && slot[old] != UNSET {
                    released.push(old);
                }
                s.index = slot[old];
            }
        });
        released.sort_unstable();
        released.dedup();
        free.extend(released.iter().map(|&old| slot[old]));
        let target = c.target as usize;
        let s = free.pop().unwrap_or_else(|| {
            slots += 1;
            slots - 1
        });
        if target < n {
            slot[target] = s;
            // Written but never read (and not the row's result): free at once.
            if last_read[target] == usize::MAX && i != last {
                free.push(s);
            }
        }
        c.target = s;
    }
    slots
}

/// A proving key's extended-domain columns on the device (fixed cosets,
/// permutation cosets, l0, l_last, l_active_row), uploaded by the first proof
/// that needs them and reused by every later proof with the key, until the key
/// is dropped. Takes a share of the backend's process-wide budget
/// (`MIDNIGHT_GPU_KEY_CACHE_MB`); a key that does not fit is read from the host
/// as before. A clone of the key starts without a copy.
#[derive(Default)]
pub(crate) struct KeyColumns(Mutex<Option<Arc<CachedKey>>>);

struct CachedKey {
    columns: DeviceColumns,
    _reservation: CacheReservation,
}

impl Clone for KeyColumns {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for KeyColumns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cached = self.0.lock().map(|c| c.is_some()).unwrap_or(false);
        f.debug_struct("KeyColumns").field("cached", &cached).finish()
    }
}

/// The key's host columns in device order: fixed cosets, permutation cosets,
/// l0, l_last, l_active_row.
fn key_host_columns<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>(
    pk: &ProvingKey<F, CS>,
) -> Vec<&[Fq]> {
    pk.fixed_cosets
        .iter()
        .chain(&pk.permutation.cosets)
        .chain([&pk.l0, &pk.l_last, &pk.l_active_row])
        .map(|p| fq(&p[..]))
        .collect()
}

/// The key's device columns, uploading them on first use if the budget allows.
fn key_columns<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>(
    pk: &ProvingKey<F, CS>,
) -> Option<Arc<CachedKey>> {
    let mut cached = pk.gpu_columns.0.lock().ok()?;
    if let Some(c) = &*cached {
        return Some(c.clone());
    }
    let host = key_host_columns(pk);
    let reservation = ge::reserve_key_cache(ge::device_bytes(pk.vk.domain.extended_len(), host.len()))?;
    match ge::upload_columns(&host) {
        Ok(columns) => {
            let c = Arc::new(CachedKey { columns, _reservation: reservation });
            *cached = Some(c.clone());
            Some(c)
        }
        Err(e) => {
            // The proof still runs, reading the key from the host.
            note_gpu_fallback("proving-key columns upload", &e.to_string());
            None
        }
    }
}

/// `F` viewed as `Fq` (the caller checked that `F` is `Fq`).
fn fq<F: PrimeField>(s: &[F]) -> &[Fq] {
    try_as_fq_slice(s).expect("F is Fq (checked by try_compute_h_poly)")
}

fn fq_scalar<F: PrimeField>(x: &F) -> Option<Fq> {
    try_as_fq_slice(std::slice::from_ref(x)).map(|s| s[0])
}

/// `GraphEvaluator<F>` viewed as `GraphEvaluator<Fq>` (`F` is `Fq`, checked).
fn as_fq_graph<F: PrimeField>(ev: &GraphEvaluator<F>) -> Option<&GraphEvaluator<Fq>> {
    try_as_fq_slice(&ev.constants)?;
    // SAFETY: the TypeId check above (try_as_fq_slice) guarantees F == Fq, so
    // GraphEvaluator<F> and GraphEvaluator<Fq> are the same type.
    Some(unsafe { &*(ev as *const GraphEvaluator<F> as *const GraphEvaluator<Fq>) })
}

/// Where the program's auxiliary columns come from: `0..3` l0, l_last,
/// l_active_row; `pk_base..product_base` the key's permutation cosets; from
/// `product_base` the proof's own columns: the permutation product of each set,
/// then per lookup its product, permuted input and permuted table, then each
/// trash polynomial.
struct Layout {
    pk_base: usize,
    product_base: usize,
    n_sets: usize,
    n_lookups: usize,
    n_aux: usize,
}

/// Compiles one circuit's quotient-numerator iteration (as `evaluate_h` runs
/// it) into a program over the kernel's sources; the same program serves every
/// circuit of the proof.
#[allow(clippy::too_many_arguments)]
fn build_program<F: WithSmallOrderMulGroup<3>>(
    evaluator: &Evaluator<F>,
    domain: &EvaluationDomain<F>,
    cs: &ConstraintSystem<F>,
    beta: F,
    n_pk: usize,
    n_sets: usize,
    n_lookups: usize,
    n_trash: usize,
) -> Option<(GraphProgram, Layout)> {
    let gates = as_fq_graph(&evaluator.custom_gates)?;
    let mut b = Builder::default();
    let one_c = b.constant(ge::fq_limbs(&Fq::ONE));
    let zero_c = b.constant(ge::fq_limbs(&Fq::ZERO));
    let pk_base = 3;
    let product_base = pk_base + n_pk;
    let lookup_base = product_base + n_sets;
    let trash_base = lookup_base + 3 * n_lookups;
    let layout = Layout { pk_base, product_base, n_sets, n_lookups, n_aux: trash_base + n_trash };
    let l0_s = b.column(src::AUX, 0, 0);
    let l_last_s = b.column(src::AUX, 1, 0);
    let l_active_s = b.column(src::AUX, 2, 0);

    // Custom gates, reading the row's running value.
    let mut value = b.inline(gates, Src::scalar(src::PREVIOUS));

    // Permutations.
    if n_sets > 0 {
        let (beta_q, g_coset, delta) = (fq_scalar(&beta)?, fq_scalar(&domain.g_coset)?, fq_scalar(&F::DELTA)?);
        let last_rotation = -((cs.blinding_factors() + 1) as i32);
        let chunk_len = cs.degree() - 2;
        let z = |b: &mut Builder, set: usize, rot: i32| b.column(src::AUX, product_base + set, rot);

        // l_0(X) * (1 - z_0(X))
        let z0 = z(&mut b, 0, 0);
        let t = b.sub(one_c, z0);
        let t = b.mul(t, l0_s);
        value = b.fold(value, t);
        // l_last(X) * (z_l(X)^2 - z_l(X))
        let zl = z(&mut b, n_sets - 1, 0);
        let sq = b.square(zl);
        let t = b.sub(sq, zl);
        let t = b.mul(t, l_last_s);
        value = b.fold(value, t);
        // l_0(X) * (z_i(X) - z_{i-1}(omega^last X)), i > 0
        for i in 1..n_sets {
            let zi = z(&mut b, i, 0);
            let zp = z(&mut b, i - 1, last_rotation);
            let t = b.sub(zi, zp);
            let t = b.mul(t, l0_s);
            value = b.fold(value, t);
        }
        // l_active(X) * (z_i(omega X) prod_j (p_j + beta s_j + gamma)
        //                - z_i(X) prod_j (p_j + delta^k beta g X + gamma)),
        // k counting the columns across the sets, X = omega^idx.
        let mut current_delta = beta_q * g_coset;
        let columns = p_columns(cs);
        for (set, cols) in columns.chunks(chunk_len).take(n_sets).enumerate() {
            // The key's permutation cosets of this set (one per column).
            let pk = n_pk.saturating_sub(set * chunk_len).min(chunk_len);
            let mut left = z(&mut b, set, 1);
            for (j, &(kind, index)) in cols.iter().enumerate().take(pk) {
                let v = b.column(kind, index, 0);
                let sigma = b.column(src::AUX, pk_base + set * chunk_len + j, 0);
                let bs = b.mul(Src::scalar(src::BETA), sigma);
                let t = b.add(v, bs);
                let t = b.add(t, Src::scalar(src::GAMMA));
                left = b.mul(left, t);
            }
            let mut right = z(&mut b, set, 0);
            for &(kind, index) in cols {
                let v = b.column(kind, index, 0);
                let d = b.constant(ge::fq_limbs(&current_delta));
                let dx = b.mul(d, Src::scalar(src::X));
                let t = b.add(v, dx);
                let t = b.add(t, Src::scalar(src::GAMMA));
                right = b.mul(right, t);
                current_delta *= delta;
            }
            let t = b.sub(left, right);
            let t = b.mul(t, l_active_s);
            value = b.fold(value, t);
        }
    }

    // Lookups (the evaluator of lookup n compresses its table expressions).
    for n in 0..n_lookups {
        let ev = evaluator.lookups.get(n).and_then(as_fq_graph)?;
        let table_value = b.inline(ev, zero_c);
        let col = |b: &mut Builder, i: usize, rot: i32| b.column(src::AUX, lookup_base + 3 * n + i, rot);
        let (zc, zn) = (col(&mut b, 0, 0), col(&mut b, 0, 1));
        let (a, a_prev, s) = (col(&mut b, 1, 0), col(&mut b, 1, -1), col(&mut b, 2, 0));
        let a_minus_s = b.sub(a, s);
        // l_0(X) * (1 - z(X))
        let t = b.sub(one_c, zc);
        let t = b.mul(t, l0_s);
        value = b.fold(value, t);
        // l_last(X) * (z(X)^2 - z(X))
        let sq = b.square(zc);
        let t = b.sub(sq, zc);
        let t = b.mul(t, l_last_s);
        value = b.fold(value, t);
        // l_active(X) * (z(omega X) (a'(X) + beta) (s'(X) + gamma) - z(X) table(X))
        let ab = b.add(a, Src::scalar(src::BETA));
        let sg = b.add(s, Src::scalar(src::GAMMA));
        let l = b.mul(zn, ab);
        let l = b.mul(l, sg);
        let r = b.mul(zc, table_value);
        let t = b.sub(l, r);
        let t = b.mul(t, l_active_s);
        value = b.fold(value, t);
        // l_0(X) * (a'(X) - s'(X))
        let t = b.mul(a_minus_s, l0_s);
        value = b.fold(value, t);
        // l_active(X) * (a'(X) - s'(X)) * (a'(X) - a'(omega^-1 X))
        let d = b.sub(a, a_prev);
        let t = b.mul(a_minus_s, d);
        let t = b.mul(t, l_active_s);
        value = b.fold(value, t);
    }

    // Trashcans: compressed expression - (1 - q) * trash.
    for n in 0..n_trash {
        let ev = evaluator.trashcans.get(n).and_then(as_fq_graph)?;
        let Expression::Fixed(query) = cs.trashcans[n].selector() else {
            return None;
        };
        let expr = b.inline(ev, zero_c);
        let q = b.column(src::FIXED, query.column_index(), 0);
        let t = b.column(src::AUX, trash_base + n, 0);
        let one_minus_q = b.sub(one_c, q);
        let r = b.mul(one_minus_q, t);
        let term = b.sub(expr, r);
        value = b.fold(value, term);
    }

    Some((b.finish(value), layout))
}

/// The permutation argument's columns as (source kind, index).
fn p_columns<F: PrimeField>(cs: &ConstraintSystem<F>) -> Vec<(u32, usize)> {
    cs.permutation
        .columns
        .iter()
        .map(|column| match column.column_type() {
            Any::Advice(_) => (src::ADVICE, column.index()),
            Any::Fixed => (src::FIXED, column.index()),
            Any::Instance => (src::INSTANCE, column.index()),
        })
        .collect()
}

/// Which columns of each kind a program reads.
struct Used {
    fixed: Vec<bool>,
    advice: Vec<bool>,
    instance: Vec<bool>,
    aux: Vec<bool>,
}

fn used_columns(program: &GraphProgram, n_fixed: usize, n_advice: usize, n_instance: usize, n_aux: usize) -> Used {
    let mut used = Used {
        fixed: vec![false; n_fixed],
        advice: vec![false; n_advice],
        instance: vec![false; n_instance],
        aux: vec![false; n_aux],
    };
    let mut calcs = program.calcs.clone();
    let mut parts = program.parts.clone();
    for c in &mut calcs {
        for_each_read(c, &mut parts, |s| {
            let set = match s.kind {
                src::FIXED => &mut used.fixed,
                src::ADVICE => &mut used.advice,
                src::INSTANCE => &mut used.instance,
                src::AUX => &mut used.aux,
                _ => return,
            };
            // An out-of-range index stays unmarked; the kernel rejects it.
            if let Some(u) = set.get_mut(s.index as usize) {
                *u = true;
            }
        });
    }
    used
}

/// The proof's polynomial behind auxiliary column `j >= layout.product_base`.
fn proof_aux_poly<'a, F: PrimeField>(
    layout: &Layout,
    j: usize,
    permutation: &'a permutation::prover::Committed<F>,
    lookups: &'a [lookup::prover::Committed<F>],
    trashcans: &'a [trash::prover::Committed<F>],
) -> Option<&'a Polynomial<F, Coeff>> {
    let r = j.checked_sub(layout.product_base)?;
    if r < layout.n_sets {
        return permutation.sets.get(r).map(|s| &s.permutation_product_poly);
    }
    let r = r - layout.n_sets;
    if r < 3 * layout.n_lookups {
        let l = lookups.get(r / 3)?;
        return Some(match r % 3 {
            0 => &l.product_poly,
            1 => &l.permuted_input_poly,
            _ => &l.permuted_table_poly,
        });
    }
    trashcans.get(r - 3 * layout.n_lookups).map(|t| &t.trash_poly)
}

/// The prover's quotient numerator (`compute_h_poly`'s result) computed on the
/// GPU from the coefficient-form trace, or `None` if the GPU path does not
/// apply or failed (the caller then computes it on the CPU).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_compute_h_poly<F, CS>(
    pk: &ProvingKey<F, CS>,
    advice_polys: &[Vec<Polynomial<F, Coeff>>],
    instance_polys: &[Vec<Polynomial<F, Coeff>>],
    lookups: &[Vec<lookup::prover::Committed<F>>],
    trashcans: &[Vec<trash::prover::Committed<F>>],
    permutations: &[permutation::prover::Committed<F>],
    challenges: &[F],
    [y, beta, gamma, theta, trash_challenge]: [F; 5],
) -> Option<Polynomial<F, ExtendedLagrangeCoeff>>
where
    F: WithSmallOrderMulGroup<3>,
    CS: PolynomialCommitmentScheme<F>,
{
    let domain = &pk.vk.domain;
    let cs = &pk.vk.cs;
    let ext_len = domain.extended_len();
    // Size first: in CPU mode this is false and nothing below runs.
    if !should_use_gpu_ntt(ext_len) || fq_scalar(&y).is_none() || !ge::is_available() {
        return None;
    }
    let n_circuits = advice_polys.len();
    if instance_polys.len() != n_circuits
        || lookups.len() != n_circuits
        || trashcans.len() != n_circuits
        || permutations.len() != n_circuits
    {
        return None;
    }
    let n_sets = permutations.first().map_or(0, |p| p.sets.len());
    let (n_lookups, n_trash) = (cs.lookups.len(), cs.trashcans.len());
    let (program, layout) = build_program(
        &pk.ev,
        domain,
        cs,
        beta,
        pk.permutation.cosets.len(),
        n_sets,
        n_lookups,
        n_trash,
    )?;
    let (n_fixed, n_pk) = (pk.fixed_cosets.len(), pk.permutation.cosets.len());
    let n_advice = advice_polys.first().map_or(0, |a| a.len());
    let n_instance = instance_polys.first().map_or(0, |i| i.len());
    let used = used_columns(&program, n_fixed, n_advice, n_instance, layout.n_aux);
    let proof_aux: Vec<usize> = (layout.product_base..layout.n_aux).filter(|&j| used.aux[j]).collect();
    let advice_used: Vec<usize> = (0..n_advice).filter(|&c| used.advice[c]).collect();
    let instance_used: Vec<usize> = (0..n_instance).filter(|&c| used.instance[c]).collect();

    // Device memory: the proof's extended columns (and the key's, if not
    // cached), the running values, the kernel's scratch and the NTT's batch.
    let key = key_columns(pk);
    let column_bytes = ge::device_bytes(ext_len, 1);
    let proof_columns = (advice_used.len() + instance_used.len() + proof_aux.len()) as u64;
    let key_columns_uploaded = if key.is_some() {
        0
    } else {
        (used.fixed.iter().filter(|u| **u).count() + used.aux[..layout.product_base].iter().filter(|u| **u).count())
            as u64
    };
    let need = column_bytes * (proof_columns + key_columns_uploaded + 1) + (512 << 20) + MEMORY_MARGIN;
    match ge::device_memory() {
        Some((_, free)) if free >= need => {}
        _ => return None,
    }

    let (Some(scalars), Some(omega), Some(g_coset)) = (
        [beta, gamma, theta, trash_challenge, y].iter().map(fq_scalar).collect::<Option<Vec<Fq>>>(),
        fq_scalar(&domain.get_extended_omega()),
        fq_scalar(&domain.g_coset),
    ) else {
        return None;
    };
    // Kernel order: beta, gamma, theta, trash challenge, y.
    let scalars: [Fq; 5] = scalars.try_into().ok()?;
    // The coset powers of coeff_to_extended: [g_coset, g_coset_inv], where the
    // domain sets g_coset_inv = g_coset^2 (g_coset is a cube root of unity).
    let zeta = [g_coset, g_coset.square()];
    let rot_scale: i32 = 1 << (domain.extended_k() - domain.k());
    let challenges = try_as_fq_slice(challenges)?;
    let key_host = key_host_columns(pk);
    let mut values = domain.empty_extended();

    for i in 0..n_circuits {
        let (advice, instance) = (&advice_polys[i], &instance_polys[i]);
        if advice.len() != n_advice
            || instance.len() != n_instance
            || permutations[i].sets.len() != n_sets
            || lookups[i].len() != n_lookups
            || trashcans[i].len() != n_trash
        {
            return None;
        }
        // This circuit's columns into the extended coset, on the device.
        let mut coeffs: Vec<&[Fq]> = Vec::new();
        coeffs.extend(advice_used.iter().map(|&c| fq(&advice[c][..])));
        coeffs.extend(instance_used.iter().map(|&c| fq(&instance[c][..])));
        for &j in &proof_aux {
            coeffs.push(fq(&proof_aux_poly(&layout, j, &permutations[i], &lookups[i], &trashcans[i])?[..]));
        }
        let extended = if coeffs.is_empty() {
            None
        } else {
            match ge::coset_extend(&coeffs, domain.extended_k(), zeta) {
                Ok(e) => Some(e),
                Err(e) => {
                    note_gpu_fallback("coset extension (quotient numerator)", &e.to_string());
                    return None;
                }
            }
        };
        let proof_col = |pos: usize| extended.as_ref().map(|e| Column::Device(e, pos));

        let key_col = |k: usize| match &key {
            Some(cached) => Column::Device(&cached.columns, k),
            None => Column::Host(key_host[k]),
        };
        let fixed: Vec<Option<Column>> =
            (0..n_fixed).map(|c| used.fixed[c].then(|| key_col(c))).collect();
        let mut advice_cols: Vec<Option<Column>> = vec![None; n_advice];
        for (pos, &c) in advice_used.iter().enumerate() {
            advice_cols[c] = proof_col(pos);
        }
        let mut instance_cols: Vec<Option<Column>> = vec![None; n_instance];
        for (pos, &c) in instance_used.iter().enumerate() {
            instance_cols[c] = proof_col(advice_used.len() + pos);
        }
        let mut aux: Vec<Option<Column>> = (0..layout.n_aux)
            .map(|j| match j {
                _ if !used.aux[j] || j >= layout.product_base => None,
                0..=2 => Some(key_col(n_fixed + n_pk + j)),
                _ => Some(key_col(n_fixed + (j - layout.pk_base))),
            })
            .collect();
        for (pos, &j) in proof_aux.iter().enumerate() {
            aux[j] = proof_col(advice_used.len() + instance_used.len() + pos);
        }

        let result = ge::eval_graph_mixed(
            &program,
            MixedColumns { fixed: &fixed, advice: &advice_cols, instance: &instance_cols, aux: &aux },
            challenges,
            scalars,
            rot_scale,
            Some(omega),
            try_as_fq_slice_mut(&mut values[..])?,
        );
        if let Err(e) = result {
            note_gpu_fallback("graph evaluation (quotient numerator)", &e.to_string());
            return None;
        }
    }
    Some(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interm(i: u32) -> Src {
        Src::indexed(src::INTERMEDIATE, i as usize)
    }

    /// Runs a program on one row with an interpreter over u64 (wrapping
    /// arithmetic stands in for the field: renaming must not change results).
    fn run(p: &GraphProgram, inputs: &[u64]) -> u64 {
        let mut inter = vec![0u64; p.num_intermediates as usize];
        let load = |s: &Src, inter: &[u64]| match s.kind {
            src::INTERMEDIATE => inter[s.index as usize],
            src::ADVICE => inputs[s.index as usize],
            _ => 7,
        };
        for c in &p.calcs {
            let r = match c.op {
                op::ADD => load(&c.a, &inter).wrapping_add(load(&c.b, &inter)),
                op::SUB => load(&c.a, &inter).wrapping_sub(load(&c.b, &inter)),
                op::MUL => load(&c.a, &inter).wrapping_mul(load(&c.b, &inter)),
                op::SQUARE => load(&c.a, &inter).wrapping_mul(load(&c.a, &inter)),
                op::HORNER => {
                    let f = load(&c.c, &inter);
                    let mut v = load(&c.a, &inter);
                    for k in 0..c.parts_len {
                        v = v.wrapping_mul(f).wrapping_add(load(&p.parts[(c.parts_offset + k) as usize], &inter));
                    }
                    v
                }
                _ => load(&c.a, &inter),
            };
            inter[c.target as usize] = r;
        }
        inter[p.calcs.last().unwrap().target as usize]
    }

    #[test]
    fn slot_allocation_preserves_results_and_reuses_slots() {
        // A chain of 40 products of pairs, summed by a final Horner over every
        // fifth of them (long-lived), as gate programs are shaped.
        let mut b = Builder::default();
        let mut horner_parts = Vec::new();
        let mut acc = Src::column(src::ADVICE, 0, 0);
        for i in 0..40 {
            let x = b.mul(acc, Src::column(src::ADVICE, i % 4, 0));
            let y = b.add(x, Src::column(src::ADVICE, (i + 1) % 4, 0));
            acc = b.sub(y, x);
            if i % 5 == 0 {
                horner_parts.push(acc);
            }
            let _dead = b.square(y);
        }
        let target = b.fresh();
        let offset = b.parts.len() as u32;
        b.parts.extend(horner_parts.iter().copied());
        b.calcs.push(Calc {
            op: op::HORNER,
            target,
            a: acc,
            c: Src::column(src::ADVICE, 3, 0),
            parts_offset: offset,
            parts_len: horner_parts.len() as u32,
            ..Default::default()
        });
        let unpacked = GraphProgram {
            calcs: b.calcs.clone(),
            parts: b.parts.clone(),
            constants: vec![],
            rotations: vec![0],
            num_intermediates: b.num_intermediates,
        };
        let packed = b.finish(interm(target));
        assert!(packed.num_intermediates < unpacked.num_intermediates / 4, "{}", packed.num_intermediates);
        for inputs in [[1u64, 2, 3, 4], [5, 0, 9, 1 << 40], [u64::MAX, 3, 7, 11]] {
            assert_eq!(run(&packed, &inputs), run(&unpacked, &inputs));
        }
    }

    #[test]
    fn read_before_write_stays_out_of_range() {
        let mut calcs = vec![Calc { op: op::STORE, target: 0, a: interm(1), ..Default::default() }];
        let slots = allocate_slots(&mut calcs, &mut [], 2);
        assert_eq!(slots, 1);
        assert!(calcs[0].a.index >= slots);
    }
}
