use ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};

use super::{super::Error, Argument};
use crate::{
    plonk::evaluation::evaluate,
    poly::{
        commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, LagrangeCoeff, Polynomial,
        ProverQuery,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::eval_polynomial,
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
    /// Returns `TrashComputed` which contains the trash polynomial in Lagrange form.
    /// Call `TrashComputed::finalize` with a commitment to complete the process.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute<'a>(
        &self,
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
        let compressed_expression = self
            .constraint_expressions
            .iter()
            .map(|expression| {
                domain.lagrange_from_vec(evaluate(
                    expression,
                    domain.n as usize,
                    1,
                    fixed_values,
                    advice_values,
                    instance_values,
                    challenges,
                ))
            })
            .fold(domain.empty_lagrange(), |acc, expression| {
                acc * trash_challenge + &expression
            });

        TrashComputed {
            trash_poly_lagrange: compressed_expression,
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
