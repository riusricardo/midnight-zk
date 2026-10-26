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

/// Whether `CS` is KZG over midnight-curves' BLS12-381 with `F = Fq`
/// (`KZGCommitmentScheme<Bls12>`), the only configuration the GPU batch commit
/// supports. When it holds, callers view `&CS::Parameters` as `&ParamsKZG<Bls12>`,
/// `Polynomial<F, _>` as `Polynomial<Fq, _>`, and the `G1Projective` results of
/// [`ParamsKZG::commit_lagrange_batch`] as `CS::Commitment`.
///
/// # Why this check is sufficient
///
/// `PolynomialCommitmentScheme` has no `'static` bound, so `CS` and its associated
/// types cannot be compared by `TypeId` without adding `CS: 'static` to upstream's
/// public prover signatures (`create_proof`, `permutation::Argument::commit`, ...).
/// `F` is `'static` (`ff::Field: 'static`) and is compared exactly. The only
/// `PolynomialCommitmentScheme` implementation is `KZGCommitmentScheme<E>`, which is
/// implemented for `F = E::Fr` with `Parameters = ParamsKZG<E>` and
/// `Commitment = E::G1`, and `Bls12` is the only midnight-curves engine whose scalar
/// field is `Fq`. So `F == Fq` already forces `CS = KZGCommitmentScheme<Bls12>`, and
/// the views above are identity reinterpretations (`Bls12::G1 = G1Projective`).
///
/// The size and alignment checks are a backstop that keeps those views in bounds.
/// They cannot tell apart a third-party scheme over `Fq` (or a third-party engine
/// with `Fr = Fq`) whose parameter and commitment types merely share these layouts:
/// such a scheme must not be combined with the `gpu` feature.
#[cfg(feature = "gpu")]
pub(crate) fn is_kzg_bls12<F, CS>() -> bool
where
    F: PrimeField + 'static,
    CS: PolynomialCommitmentScheme<F>,
{
    crate::gpu_accel::is_fq::<F>()
        && size_of::<CS::Parameters>() == size_of::<ParamsKZG<Bls12>>()
        && align_of::<CS::Parameters>() == align_of::<ParamsKZG<Bls12>>()
        && size_of::<CS::Commitment>() == size_of::<G1Projective>()
        && align_of::<CS::Commitment>() == align_of::<G1Projective>()
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
/// - For multiple polynomials, GPU pipelining provides 15-40% speedup
/// - The GPU threshold (K≥15, 32768 points) is checked internally
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
    
    // Check if we're using KZG with BLS12-381 (midnight_curves) - the GPU-supported configuration
    // AND if GPU batching would be beneficial for this size
    let individual_size = polys[0].len();
    let batch_count = polys.len();
    let gpu_beneficial = should_use_gpu_batch(individual_size, batch_count);
    
    if gpu_beneficial && is_kzg_bls12::<F, CS>() {
        #[cfg(feature = "trace-kzg")]
        eprintln!("[BATCH_COMMIT] GPU pipelining {} polynomials", polys.len());
        
        // SAFETY: `is_kzg_bls12` holds, so CS is `KZGCommitmentScheme<Bls12>` (see its
        // docs for why the check is sufficient):
        // 1. CS::Parameters is ParamsKZG<Bls12>
        // 2. F is midnight_curves::Fq, so Polynomial<F, L> is Polynomial<Fq, L>
        // 3. CS::Commitment is Bls12's G1 = G1Projective, the type returned by
        //    commit_lagrange_batch
        unsafe {
            let kzg_params = &*(params as *const CS::Parameters as *const ParamsKZG<Bls12>);
            let poly_refs: Vec<&Polynomial<Fq, LagrangeCoeff>> = 
                polys.iter()
                    .map(|p| &*(p as *const Polynomial<F, LagrangeCoeff> 
                               as *const Polynomial<Fq, LagrangeCoeff>))
                    .collect();
            
            let batch_commits = kzg_params.commit_lagrange_batch(&poly_refs);
            
            // Transmute back to CS::Commitment (G1Projective, see SAFETY above)
            batch_commits.into_iter()
                .map(|c| std::mem::transmute_copy(&c))
                .collect()
        }
    } else {
        // For CPU path, just call commit_lagrange directly - no batching overhead
        polys.iter()
            .map(|poly| CS::commit_lagrange(params, poly))
            .collect()
    }
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
    
    // Check if we're using KZG with BLS12-381 (midnight_curves)
    // AND if GPU batching would be beneficial
    let individual_size = polys[0].len();
    let batch_count = polys.len();
    let gpu_beneficial = should_use_gpu_batch(individual_size, batch_count);
    
    if gpu_beneficial && is_kzg_bls12::<F, CS>() {
        #[cfg(feature = "trace-kzg")]
        eprintln!("[BATCH_COMMIT] GPU pipelining {} polynomials (refs)", polys.len());
        
        // SAFETY: as in `batch_commit`: `is_kzg_bls12` holds, so CS::Parameters is
        // ParamsKZG<Bls12>, F is Fq and CS::Commitment is G1Projective.
        unsafe {
            let kzg_params = &*(params as *const CS::Parameters as *const ParamsKZG<Bls12>);
            let poly_refs: Vec<&Polynomial<Fq, LagrangeCoeff>> = 
                polys.iter()
                    .map(|p| &*(*p as *const Polynomial<F, LagrangeCoeff> 
                               as *const Polynomial<Fq, LagrangeCoeff>))
                    .collect();
            
            let batch_commits = kzg_params.commit_lagrange_batch(&poly_refs);
            
            batch_commits.into_iter()
                .map(|c| std::mem::transmute_copy(&c))
                .collect()
        }
    } else {
        // For CPU path, just call commit_lagrange directly - no batching overhead
        polys.iter()
            .map(|poly| CS::commit_lagrange(params, *poly))
            .collect()
    }
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
