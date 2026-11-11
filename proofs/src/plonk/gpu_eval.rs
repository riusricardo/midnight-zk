//! GPU evaluation of the quotient numerator's custom gates.
//!
//! [`try_custom_gates`] runs the custom-gate `GraphEvaluator` on every row of the
//! extended domain with the CUDA backend's graph-evaluation kernel (design: the
//! backend's `docs/gpu-integration/GPU_QUOTIENT_EVALUATION.md`). It does so only
//! when the field is midnight's BLS12-381 scalar field, the GPU path is chosen
//! for this size (`should_use_gpu_ntt`: never in CPU mode, so CPU mode does not
//! touch the GPU) and the installed backend has the kernel. Each row's value is
//! bit-identical to the CPU loop's. On a GPU failure it reports the fallback
//! (fatal under `MIDNIGHT_GPU_STRICT`), restores `values` and returns `false`,
//! and the caller runs the CPU loop.

use ff::PrimeField;
use midnight_bls12_381_cuda::config::note_gpu_fallback;

use super::evaluation::{Calculation, GraphEvaluator, ValueSource};
use crate::{
    gpu_accel::{
        backend::{
            curves::Fq,
            graph_eval::{self as ge, op, src, Calc, GraphColumns, GraphProgram, Src},
        },
        should_use_gpu_ntt, try_as_fq_slice, try_as_fq_slice_mut,
    },
    poly::{Polynomial, PolynomialRepresentation},
};

fn source(v: &ValueSource) -> Src {
    match *v {
        ValueSource::Constant(i) => Src::indexed(src::CONSTANT, i),
        ValueSource::Intermediate(i) => Src::indexed(src::INTERMEDIATE, i),
        ValueSource::Fixed(c, r) => Src::column(src::FIXED, c, r),
        ValueSource::Advice(c, r) => Src::column(src::ADVICE, c, r),
        ValueSource::Instance(c, r) => Src::column(src::INSTANCE, c, r),
        ValueSource::Challenge(i) => Src::indexed(src::CHALLENGE, i),
        ValueSource::Beta() => Src::scalar(src::BETA),
        ValueSource::Gamma() => Src::scalar(src::GAMMA),
        ValueSource::Theta() => Src::scalar(src::THETA),
        ValueSource::TrashChallenge() => Src::scalar(src::TRASH_CHALLENGE),
        ValueSource::Y() => Src::scalar(src::Y),
        ValueSource::PreviousValue() => Src::scalar(src::PREVIOUS),
    }
}

/// The evaluator in the kernel's encoding; `None` unless `F` is `Fq`.
fn program<F: PrimeField>(ev: &GraphEvaluator<F>) -> Option<GraphProgram> {
    let constants = try_as_fq_slice(&ev.constants)?.iter().map(ge::fq_limbs).collect();
    let mut parts = Vec::new();
    let calcs = ev
        .calculations
        .iter()
        .map(|info| {
            let mut c = Calc { target: info.target as u32, ..Default::default() };
            match &info.calculation {
                Calculation::Add(a, b) => (c.op, c.a, c.b) = (op::ADD, source(a), source(b)),
                Calculation::Sub(a, b) => (c.op, c.a, c.b) = (op::SUB, source(a), source(b)),
                Calculation::Mul(a, b) => (c.op, c.a, c.b) = (op::MUL, source(a), source(b)),
                Calculation::Square(a) => (c.op, c.a) = (op::SQUARE, source(a)),
                Calculation::Double(a) => (c.op, c.a) = (op::DOUBLE, source(a)),
                Calculation::Negate(a) => (c.op, c.a) = (op::NEGATE, source(a)),
                Calculation::Store(a) => (c.op, c.a) = (op::STORE, source(a)),
                Calculation::Horner(start, ps, factor) => {
                    (c.op, c.a, c.c) = (op::HORNER, source(start), source(factor));
                    c.parts_offset = parts.len() as u32;
                    c.parts_len = ps.len() as u32;
                    parts.extend(ps.iter().map(source));
                }
            }
            c
        })
        .collect();
    Some(GraphProgram {
        calcs,
        parts,
        constants,
        rotations: ev.rotations.clone(),
        num_intermediates: ev.num_intermediates as u32,
    })
}

fn fq_columns<F: PrimeField, B: PolynomialRepresentation>(
    polys: &[Polynomial<F, B>],
) -> Option<Vec<Option<&[Fq]>>> {
    polys.iter().map(|p| try_as_fq_slice(&p[..]).map(Some)).collect()
}

fn fq_scalar<F: PrimeField>(x: &F) -> Option<Fq> {
    try_as_fq_slice(std::slice::from_ref(x)).map(|s| s[0])
}

/// Evaluates the custom gates on the GPU into `values` (each row's previous
/// value on entry). Returns `false`, with `values` unchanged, if the GPU path
/// does not apply or failed; the caller then runs the CPU loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_custom_gates<F: PrimeField, B: PolynomialRepresentation>(
    ev: &GraphEvaluator<F>,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
    scalars: [&F; 5],
    rot_scale: i32,
    values: &mut [F],
) -> bool {
    // Size first: in CPU mode this is false and nothing below runs.
    if !should_use_gpu_ntt(values.len()) || try_as_fq_slice(values).is_none() || !ge::is_available() {
        return false;
    }
    let (Some(program), Some(fixed), Some(advice), Some(instance), Some(challenges)) = (
        program(ev),
        fq_columns(fixed),
        fq_columns(advice),
        fq_columns(instance),
        try_as_fq_slice(challenges),
    ) else {
        return false;
    };
    let Some(scalars) = scalars.iter().map(|s| fq_scalar(*s)).collect::<Option<Vec<Fq>>>() else {
        return false;
    };
    let scalars: [Fq; 5] = scalars.try_into().expect("five scalars");
    let values = try_as_fq_slice_mut(values).expect("F is Fq (checked above)");
    // A failed copy-back could leave rows half written; the CPU loop reads each
    // row's previous value, so it must start from the original values.
    let backup = values.to_vec();
    match ge::eval_graph(
        &program,
        GraphColumns { fixed: &fixed, advice: &advice, instance: &instance },
        challenges,
        scalars,
        rot_scale,
        values,
    ) {
        Ok(()) => true,
        Err(e) => {
            note_gpu_fallback("graph evaluation (custom gates)", &e.to_string());
            values.copy_from_slice(&backup);
            false
        }
    }
}
