use ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};

use super::{super::Error, Argument};
use crate::{
    plonk::evaluation::GraphEvaluator,
    poly::{
        commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, LagrangeCoeff, Polynomial,
        ProverQuery,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::{eval_polynomial, parallelize},
};

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Committed<F: PrimeField> {
    pub(crate) trash_poly: Polynomial<F, Coeff>,
}

/// Intermediate state after computing the trash polynomial but before commitment.
/// This enables batch committing multiple trash arguments together.
#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct TrashComputed<F: PrimeField> {
    /// Trash polynomial in Lagrange form (ready for commitment)
    pub(crate) trash_poly_lagrange: Polynomial<F, LagrangeCoeff>,
}

pub(crate) struct Evaluated<F: PrimeField>(Committed<F>);

impl<F: WithSmallOrderMulGroup<3> + Ord> Argument<F> {
    /// Compute the trash polynomial without committing.
    /// This enables batch committing multiple trash arguments together.
    ///
    /// At every row the constraint expressions are folded with the trash
    /// challenge in Horner form, `((e_0 * c + e_1) * c + ...)`, which is what
    /// `graph` computes: this argument's graph in the proving key's
    /// `Evaluator` (the quotient evaluates the same graph on the extended
    /// domain). One pass over the rows evaluates every expression, with the
    /// graph's shared subexpressions, instead of one pass and one polynomial
    /// per expression; field arithmetic is exact, so the values are the same.
    ///
    /// Returns `TrashComputed` which contains the trash polynomial in Lagrange form.
    /// Call `TrashComputed::finalize` with a commitment to complete the process.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute<'a>(
        &self,
        graph: &GraphEvaluator<F>,
        domain: &EvaluationDomain<F>,
        trash_challenge: F,
        advice_values: &'a [Polynomial<F, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<F, LagrangeCoeff>],
        instance_values: &'a [Polynomial<F, LagrangeCoeff>],
        challenges: &'a [F],
    ) -> TrashComputed<F>
    where
        F: FromUniformBytes<64>,
    {
        let isize = domain.n as i32;
        let mut values = domain.empty_lagrange();
        parallelize(&mut values, |values, start| {
            let mut data = graph.instance();
            for (i, value) in values.iter_mut().enumerate() {
                *value = graph.evaluate(
                    &mut data,
                    fixed_values,
                    advice_values,
                    instance_values,
                    challenges,
                    &F::ZERO,
                    &F::ZERO,
                    &F::ZERO,
                    &trash_challenge,
                    &F::ZERO,
                    &F::ZERO,
                    start + i,
                    1,
                    isize,
                );
            }
        });

        TrashComputed {
            trash_poly_lagrange: values,
        }
    }

    /// Upstream-compatible single-trashcan commit: [`Self::compute`],
    /// `CS::commit_lagrange`, then [`TrashComputed::finalize`]. Kept for the
    /// benchmark prover (`plonk/bench/prover.rs`), which commits each
    /// trashcan on its own.
    #[cfg(feature = "bench-internal")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit<'a, 'params: 'a, CS, T>(
        &self,
        graph: &GraphEvaluator<F>,
        params: &'params CS::Parameters,
        domain: &EvaluationDomain<F>,
        trash_challenge: F,
        advice_values: &'a [Polynomial<F, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<F, LagrangeCoeff>],
        instance_values: &'a [Polynomial<F, LagrangeCoeff>],
        challenges: &'a [F],
        transcript: &mut T,
    ) -> Result<Committed<F>, Error>
    where
        F: FromUniformBytes<64>,
        CS: PolynomialCommitmentScheme<F>,
        CS::Commitment: Hashable<T::Hash>,
        T: Transcript,
    {
        let trash = self.compute(
            graph,
            domain,
            trash_challenge,
            advice_values,
            fixed_values,
            instance_values,
            challenges,
        );
        let trash_commitment = CS::commit_lagrange(params, &trash.trash_poly_lagrange);
        trash.finalize::<CS, T>(domain, trash_commitment, transcript)
    }
}

impl<F: PrimeField> TrashComputed<F> {
    /// Finalize the trash argument with a pre-computed commitment.
    /// This is used when batch committing multiple trash arguments.
    pub(crate) fn finalize<CS: PolynomialCommitmentScheme<F>, T: Transcript>(
        self,
        domain: &EvaluationDomain<F>,
        commitment: CS::Commitment,
        transcript: &mut T,
    ) -> Result<Committed<F>, Error>
    where
        F: WithSmallOrderMulGroup<3>,
        CS::Commitment: Hashable<T::Hash>,
    {
        let trash_poly = domain.lagrange_to_coeff(self.trash_poly_lagrange);
        transcript.write(&commitment)?;
        Ok(Committed { trash_poly })
    }
}

impl<F: WithSmallOrderMulGroup<3>> Committed<F> {
    pub(crate) fn evaluate<T>(self, x: F, transcript: &mut T) -> Result<Evaluated<F>, Error>
    where
        F: Hashable<T::Hash>,
        T: Transcript,
    {
        let trash_eval = eval_polynomial(&self.trash_poly, x);
        transcript.write(&trash_eval)?;

        Ok(Evaluated(self))
    }
}

impl<F: WithSmallOrderMulGroup<3>> Evaluated<F> {
    pub(crate) fn open(&self, x: F) -> impl Iterator<Item = ProverQuery<'_, F>> + Clone {
        vec![ProverQuery {
            point: x,
            poly: &self.0.trash_poly,
        }]
        .into_iter()
    }
}
