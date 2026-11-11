//! GPU-accelerated batch polynomial commitment
//!
//! This module provides batch commitment functionality that enables GPU pipelining
//! for multiple polynomial commitments. When using KZG with BLS12-381 curves and
//! the GPU feature is enabled, multiple commits are launched asynchronously to
//! overlap GPU computation with memory transfers.
//!
//! # Usage
//!
//! The module uses the type-safe dispatch helpers from `gpu_accel` to automatically
//! route to GPU or CPU based on type and size.
//!
//! # Example
//!
//! ```rust,ignore
//! use midnight_proofs::poly::batch_commit::batch_commit;
//!
//! // Collect polynomials to commit
//! let polys = vec![poly1, poly2, poly3, poly4];
//!
//! // Batch commit (GPU pipelining if available, else sequential)
//! let commitments = batch_commit::<F, CS>(params, &polys);
//! ```

use ff::PrimeField;
use crate::poly::{LagrangeCoeff, Polynomial};
use crate::poly::commitment::PolynomialCommitmentScheme;

#[cfg(feature = "gpu")]
use std::mem::{size_of, align_of};
#[cfg(feature = "gpu")]
use crate::gpu_accel::should_use_gpu_batch;
#[cfg(feature = "gpu")]
use crate::poly::kzg::params::ParamsKZG;
#[cfg(feature = "gpu")]
use midnight_curves::{Bls12, Fq, G1Projective};

/// Commits `polys` with [`ParamsKZG::commit_lagrange_batch`] (GPU pipelining)
/// when `CS` is KZG over midnight BLS12-381, the GPU-supported configuration.
/// Returns `None` for any other configuration; the caller then commits the
/// polynomials one by one with `CS::commit_lagrange`.
///
/// # How `CS` is identified
///
/// `PolynomialCommitmentScheme` has no `'static` bound, and neither do its
/// callers up to the public `create_proof`, so `TypeId` (which needs
/// `'static`) only applies to `F` (`PrimeField: 'static`). The check is:
/// - `F == midnight_curves::Fq`, exactly (`TypeId`);
/// - `CS::Parameters` has the size and alignment of `ParamsKZG<Bls12>`, and
///   `CS::Commitment` those of `G1Projective`.
///
/// This identifies `KZGCommitmentScheme<Bls12>` among every scheme
/// midnight-proofs defines: its only `PolynomialCommitmentScheme`
/// implementation is `KZGCommitmentScheme<E>` (`Parameters = ParamsKZG<E>`,
/// `Commitment = E::G1`), and `F = E::Fr = Fq` singles out `E = Bls12` among
/// the engines of midnight-curves (`Bn256`, the only other one, has its own
/// `Fr`). For that scheme `Parameters` is exactly `ParamsKZG<Bls12>` and
/// `Commitment` exactly `G1Projective`, so the casts below are identities.
///
/// It is not a proof of type identity: a third-party scheme over `Fq` (or an
/// `Engine` with `Fr = Fq`) whose types merely share these layouts would pass
/// and make the casts unsound. No such implementation exists in midnight-zk or
/// the Midnight ledger. Making the check exact needs a `'static` bound on `CS`
/// (a breaking change to `create_proof`) or a batch-commit method on
/// `PolynomialCommitmentScheme`.
#[cfg(feature = "gpu")]
pub(crate) fn commit_lagrange_batch_if_kzg_bls12<F, CS>(
    params: &CS::Parameters,
    polys: &[&Polynomial<F, LagrangeCoeff>],
) -> Option<Vec<CS::Commitment>>
where
    F: PrimeField + 'static,
    CS: PolynomialCommitmentScheme<F>,
{
    use std::any::Any;

    let is_kzg_bls12 = crate::gpu_accel::is_fq::<F>()
        && size_of::<CS::Parameters>() == size_of::<ParamsKZG<Bls12>>()
        && align_of::<CS::Parameters>() == align_of::<ParamsKZG<Bls12>>()
        && size_of::<CS::Commitment>() == size_of::<G1Projective>()
        && align_of::<CS::Commitment>() == align_of::<G1Projective>();
    if !is_kzg_bls12 {
        return None;
    }

    #[cfg(feature = "trace-kzg")]
    eprintln!("[BATCH_COMMIT] KZG/BLS12-381 batch commit of {} polynomials", polys.len());

    // `F` is `Fq` (TypeId), so this checked downcast is the identity.
    let poly_refs: Vec<&Polynomial<Fq, LagrangeCoeff>> = polys
        .iter()
        .map(|p| {
            (*p as &dyn Any)
                .downcast_ref::<Polynomial<Fq, LagrangeCoeff>>()
                .expect("F is Fq (TypeId-checked above)")
        })
        .collect();

    // SAFETY: `CS` is `KZGCommitmentScheme<Bls12>` (see "How `CS` is
    // identified"), so `CS::Parameters` is `ParamsKZG<Bls12>` and
    // `CS::Commitment` is `<Bls12 as Engine>::G1 = G1Projective`, the element
    // type `commit_lagrange_batch` returns: both casts are identities.
    unsafe {
        let kzg_params = &*(params as *const CS::Parameters as *const ParamsKZG<Bls12>);
        let commitments: Vec<G1Projective> = kzg_params.commit_lagrange_batch(&poly_refs);
        Some(
            commitments
                .iter()
                .map(|c| std::mem::transmute_copy::<G1Projective, CS::Commitment>(c))
                .collect(),
        )
    }
}

/// Batch commit multiple Lagrange polynomials with GPU pipelining when available.
///
/// This function automatically uses GPU batch commits when:
/// 1. The `gpu` feature is enabled
/// 2. Using KZG with BLS12-381 (midnight_curves::Bls12)
/// 3. There are 2+ polynomials to commit
///
/// Otherwise, falls back to sequential individual commits.
///
/// # Arguments
///
/// * `params` - Commitment scheme parameters
/// * `polys` - Slice of polynomials in Lagrange form to commit
///
/// # Returns
///
/// Vector of commitments in the same order as input polynomials.
///
/// # Performance Notes
///
/// - For single polynomials, this is equivalent to calling `CS::commit_lagrange` directly
/// - For multiple polynomials, the GPU path pipelines the uploads and MSMs
/// - The GPU/CPU size threshold (the calibrated MSM crossover) is checked internally
#[cfg(feature = "gpu")]
pub fn batch_commit<F, CS>(
    params: &CS::Parameters,
    polys: &[Polynomial<F, LagrangeCoeff>],
) -> Vec<CS::Commitment>
where
    F: PrimeField + 'static,
    CS: PolynomialCommitmentScheme<F>,
{
    if polys.is_empty() {
        return vec![];
    }
    
    // For single polynomial, just commit directly
    if polys.len() == 1 {
        return vec![CS::commit_lagrange(params, &polys[0])];
    }
    
    // Use the KZG/BLS12-381 batch (GPU pipelining) only if GPU batching
    // would be beneficial for this size
    let individual_size = polys[0].len();
    let batch_count = polys.len();
    let gpu_beneficial = should_use_gpu_batch(individual_size, batch_count);
    
    if gpu_beneficial {
        let poly_refs: Vec<&Polynomial<F, LagrangeCoeff>> = polys.iter().collect();
        if let Some(commitments) = commit_lagrange_batch_if_kzg_bls12::<F, CS>(params, &poly_refs) {
            return commitments;
        }
    }

    // For CPU path, just call commit_lagrange directly - no batching overhead
    polys.iter()
        .map(|poly| CS::commit_lagrange(params, poly))
        .collect()
}

/// Non-GPU version that just does sequential commits
#[cfg(not(feature = "gpu"))]
pub fn batch_commit<F, CS>(
    params: &CS::Parameters,
    polys: &[Polynomial<F, LagrangeCoeff>],
) -> Vec<CS::Commitment>
where
    F: PrimeField + 'static,
    CS: PolynomialCommitmentScheme<F>,
{
    polys.iter()
        .map(|poly| CS::commit_lagrange(params, poly))
        .collect()
}

/// Batch commit polynomial references with GPU pipelining when available.
///
/// Same as [`batch_commit`] but accepts references instead of owned polynomials.
/// More efficient when polynomials are already stored elsewhere.
#[cfg(feature = "gpu")]
pub fn batch_commit_refs<F, CS>(
    params: &CS::Parameters,
    polys: &[&Polynomial<F, LagrangeCoeff>],
) -> Vec<CS::Commitment>
where
    F: PrimeField + 'static,
    CS: PolynomialCommitmentScheme<F>,
{
    if polys.is_empty() {
        return vec![];
    }
    
    // For single polynomial, just commit directly
    if polys.len() == 1 {
        return vec![CS::commit_lagrange(params, polys[0])];
    }
    
    // Use the KZG/BLS12-381 batch (GPU pipelining) only if GPU batching
    // would be beneficial
    let individual_size = polys[0].len();
    let batch_count = polys.len();
    let gpu_beneficial = should_use_gpu_batch(individual_size, batch_count);
    
    if gpu_beneficial {
        if let Some(commitments) = commit_lagrange_batch_if_kzg_bls12::<F, CS>(params, polys) {
            return commitments;
        }
    }

    // For CPU path, just call commit_lagrange directly - no batching overhead
    polys.iter()
        .map(|poly| CS::commit_lagrange(params, *poly))
        .collect()
}

/// Non-GPU version
#[cfg(not(feature = "gpu"))]
pub fn batch_commit_refs<F, CS>(
    params: &CS::Parameters,
    polys: &[&Polynomial<F, LagrangeCoeff>],
) -> Vec<CS::Commitment>
where
    F: PrimeField + 'static,
    CS: PolynomialCommitmentScheme<F>,
{
    polys.iter()
        .map(|poly| CS::commit_lagrange(params, *poly))
        .collect()
}

#[cfg(test)]
mod tests {
    // Tests would require setting up full KZG params which is done in integration tests
    // See proofs/tests/batch_commit_test.rs for full tests
}
