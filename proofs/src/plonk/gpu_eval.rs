//! GPU evaluation of the quotient numerator.
//!
//! [`try_quotient_iteration`] compiles one circuit's iteration of
//! `Evaluator::evaluate_h` (custom gates, permutation, lookups, trash) into a
//! single straight-line program and runs it on every row of the extended domain
//! with the CUDA backend's graph-evaluation kernel (design: the backend's
//! `docs/gpu-integration/GPU_QUOTIENT_EVALUATION.md`). Field arithmetic is
//! exact, so each row's value equals the CPU loop's bit for bit even where the
//! program groups the operations differently. It runs only when the field is
//! midnight's BLS12-381 scalar field, the GPU path is chosen for this size
//! (`should_use_gpu_ntt`: never in CPU mode, so CPU mode does not touch the
//! GPU) and the installed backend has the kernel. On a GPU failure it reports
//! the fallback (fatal under `MIDNIGHT_GPU_STRICT`), restores `values` and
//! returns `false`, and the caller runs the CPU loops.

use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use midnight_bls12_381_cuda::config::note_gpu_fallback;

use super::{
    evaluation::{Calculation, Evaluator, GraphEvaluator, ValueSource},
    lookup, permutation, trash, Any, ConstraintSystem, Expression,
};
use crate::{
    gpu_accel::{
        backend::{
            curves::Fq,
            graph_eval::{self as ge, op, src, Calc, GraphColumns, GraphProgram, Limbs, Src},
        },
        should_use_gpu_ntt, try_as_fq_slice, try_as_fq_slice_mut,
    },
    poly::{EvaluationDomain, Polynomial, PolynomialRepresentation},
};

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

    /// The program, whose last calculation writes `result` (the row's value).
    fn finish(mut self, result: Src) -> GraphProgram {
        let last_writes_result = self.calcs.last().is_some_and(|c| {
            result.kind == src::INTERMEDIATE && c.target == result.index
        });
        if !last_writes_result {
            let target = self.fresh();
            self.calcs.push(Calc { op: op::STORE, target, a: result, ..Default::default() });
        }
        GraphProgram {
            calcs: self.calcs,
            parts: self.parts,
            constants: self.constants,
            rotations: self.rotations,
            num_intermediates: self.num_intermediates.max(1),
        }
    }
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

/// Runs one circuit's iteration of `evaluate_h` on the GPU into `values` (the
/// running numerator on entry). Returns `false`, with `values` unchanged, if
/// the GPU path does not apply or failed; the caller then runs the CPU loops.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_quotient_iteration<F, B>(
    evaluator: &Evaluator<F>,
    domain: &EvaluationDomain<F>,
    cs: &ConstraintSystem<F>,
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    fixed: &[Polynomial<F, B>],
    challenges: &[F],
    [y, beta, gamma, theta, trash_challenge]: [F; 5],
    lookups: &[lookup::prover::Committed<F>],
    trashcans: &[trash::prover::Committed<F>],
    permutation: &permutation::prover::Committed<F>,
    [l0, l_last, l_active_row]: [&Polynomial<F, B>; 3],
    permutation_pk_cosets: &[Polynomial<F, B>],
    values: &mut [F],
) -> bool
where
    F: WithSmallOrderMulGroup<3>,
    B: PolynomialRepresentation,
{
    // Size first: in CPU mode this is false and nothing below runs.
    if !should_use_gpu_ntt(values.len()) || try_as_fq_slice(values).is_none() || !ge::is_available() {
        return false;
    }
    let Some(gates) = as_fq_graph(&evaluator.custom_gates) else {
        return false;
    };
    // As in evaluate_h: rotation r reads row (idx + r * rot_scale) mod size.
    let rot_scale: i32 = 1 << (B::k(domain) - domain.k());
    let mut b = Builder::default();
    let one_c = b.constant(ge::fq_limbs(&Fq::ONE));
    let zero_c = b.constant(ge::fq_limbs(&Fq::ZERO));

    // The cosets the CPU loops compute from coefficient form (on the GPU NTT
    // where that applies), computed up front here.
    let sets = &permutation.sets;
    let product_cosets: Vec<Polynomial<F, B>> =
        sets.iter().map(|set| B::coeff_to_self(domain, set.permutation_product_poly.clone())).collect();
    let lookup_cosets: Vec<[Polynomial<F, B>; 3]> = lookups
        .iter()
        .map(|l| {
            [
                B::coeff_to_self(domain, l.product_poly.clone()),
                B::coeff_to_self(domain, l.permuted_input_poly.clone()),
                B::coeff_to_self(domain, l.permuted_table_poly.clone()),
            ]
        })
        .collect();
    let trash_cosets: Vec<Polynomial<F, B>> =
        trashcans.iter().map(|t| B::coeff_to_self(domain, t.trash_poly.clone())).collect();

    // Auxiliary columns: l0, l_last, l_active_row, the proving key's
    // permutation cosets, the permutation product cosets, per lookup its
    // product / permuted input / permuted table cosets, the trash cosets.
    let mut aux_cols: Vec<&[F]> = vec![&l0[..], &l_last[..], &l_active_row[..]];
    let pk_base = aux_cols.len();
    aux_cols.extend(permutation_pk_cosets.iter().map(|p| &p[..]));
    let product_base = aux_cols.len();
    aux_cols.extend(product_cosets.iter().map(|p| &p[..]));
    let lookup_base = aux_cols.len();
    aux_cols.extend(lookup_cosets.iter().flat_map(|c| c.iter().map(|p| &p[..])));
    let trash_base = aux_cols.len();
    aux_cols.extend(trash_cosets.iter().map(|p| &p[..]));
    let l0_s = b.column(src::AUX, 0, 0);
    let l_last_s = b.column(src::AUX, 1, 0);
    let l_active_s = b.column(src::AUX, 2, 0);

    // Custom gates, reading the row's running value.
    let mut value = b.inline(gates, Src::scalar(src::PREVIOUS));

    // Permutations.
    if !sets.is_empty() {
        let (Some(beta_q), Some(g_coset), Some(delta)) =
            (fq_scalar(&beta), fq_scalar(&B::g_coset(domain)), fq_scalar(&F::DELTA))
        else {
            return false;
        };
        let last_rotation = -((cs.blinding_factors() + 1) as i32);
        let chunk_len = cs.degree() - 2;
        let z = |b: &mut Builder, set: usize, rot: i32| b.column(src::AUX, product_base + set, rot);

        // l_0(X) * (1 - z_0(X))
        let z0 = z(&mut b, 0, 0);
        let t = b.sub(one_c, z0);
        let t = b.mul(t, l0_s);
        value = b.fold(value, t);
        // l_last(X) * (z_l(X)^2 - z_l(X))
        let zl = z(&mut b, sets.len() - 1, 0);
        let sq = b.square(zl);
        let t = b.sub(sq, zl);
        let t = b.mul(t, l_last_s);
        value = b.fold(value, t);
        // l_0(X) * (z_i(X) - z_{i-1}(omega^last X)), i > 0
        for i in 1..sets.len() {
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
        for (set, (cols, pk)) in columns
            .chunks(chunk_len)
            .zip(permutation_pk_cosets.chunks(chunk_len).map(|c| c.len()))
            .take(sets.len())
            .enumerate()
        {
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
    for n in 0..lookups.len() {
        let Some(ev) = evaluator.lookups.get(n).and_then(as_fq_graph) else {
            return false;
        };
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
    for n in 0..trashcans.len() {
        let Some(ev) = evaluator.trashcans.get(n).and_then(as_fq_graph) else {
            return false;
        };
        let Expression::Fixed(query) = cs.trashcans[n].selector() else {
            return false;
        };
        let expr = b.inline(ev, zero_c);
        let q = b.column(src::FIXED, query.column_index(), 0);
        let t = b.column(src::AUX, trash_base + n, 0);
        let one_minus_q = b.sub(one_c, q);
        let r = b.mul(one_minus_q, t);
        let term = b.sub(expr, r);
        value = b.fold(value, term);
    }

    let program = b.finish(value);
    run(
        program,
        fixed,
        advice,
        instance,
        &aux_cols,
        challenges,
        [beta, gamma, theta, trash_challenge, y],
        B::omega(domain),
        rot_scale,
        values,
    )
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

fn cols<F: PrimeField, B: PolynomialRepresentation>(ps: &[Polynomial<F, B>]) -> Option<Vec<Option<&[Fq]>>> {
    ps.iter().map(|p| try_as_fq_slice(&p[..]).map(Some)).collect()
}

#[allow(clippy::too_many_arguments)]
fn run<F: PrimeField, B: PolynomialRepresentation>(
    program: GraphProgram,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    aux: &[&[F]],
    challenges: &[F],
    scalars: [F; 5],
    omega: F,
    rot_scale: i32,
    values: &mut [F],
) -> bool {
    let (Some(fixed), Some(advice), Some(instance), Some(challenges), Some(omega)) =
        (cols(fixed), cols(advice), cols(instance), try_as_fq_slice(challenges), fq_scalar(&omega))
    else {
        return false;
    };
    let Some(aux) = aux.iter().map(|c| try_as_fq_slice(c).map(Some)).collect::<Option<Vec<_>>>() else {
        return false;
    };
    let Some(scalars) = scalars.iter().map(fq_scalar).collect::<Option<Vec<Fq>>>() else {
        return false;
    };
    // Kernel order: beta, gamma, theta, trash challenge, y.
    let scalars: [Fq; 5] = scalars.try_into().expect("five scalars");
    let values = try_as_fq_slice_mut(values).expect("F is Fq (checked by the caller)");
    // A failed copy-back could leave rows half written; the CPU loops read the
    // running value, so they must start from the original values.
    let backup = values.to_vec();
    match ge::eval_graph(
        &program,
        GraphColumns { fixed: &fixed, advice: &advice, instance: &instance, aux: &aux },
        challenges,
        scalars,
        rot_scale,
        Some(omega),
        values,
    ) {
        Ok(()) => true,
        Err(e) => {
            note_gpu_fallback("graph evaluation (quotient numerator)", &e.to_string());
            values.copy_from_slice(&backup);
            false
        }
    }
}
