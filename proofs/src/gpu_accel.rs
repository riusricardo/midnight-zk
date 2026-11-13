/*
 * Copyright (C) 2026 Midnight Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! GPU Acceleration Bridge
//!
//! This module provides a thin integration layer between midnight-proofs and
//! the GPU acceleration backend. It re-exports everything GPU-related so that
//! the rest of midnight-proofs only needs to import from this module.
//!
//! # Design Goals
//!
//! 1. **Single Import Point**: All GPU functionality flows through `gpu_accel`
//! 2. **No Direct Backend Imports**: Other modules should NOT import from
//!    `midnight_bls12_381_cuda` or `icicle_*` directly
//! 3. **Minimal Coupling**: Makes it easy to swap backends in the future
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::gpu_accel::{should_use_gpu, PrecomputedBases, GpuMsmContext};
//!
//! if should_use_gpu(size) {
//!     let ctx = get_msm_context();
//!     let result = ctx.msm(scalars, bases)?;
//! }
//! ```
//!
//! # Type-Safe Dispatch
//!
//! For generic code that needs to dispatch to GPU based on type:
//!
//! ```rust,ignore
//! use crate::gpu_accel::{dispatch_msm, with_fq_slice, is_g1_affine};
//!
//! // Option 1: High-level dispatch (handles everything)
//! let result = dispatch_msm::<C, _, _>(
//!     coeffs,
//!     |fq_coeffs| gpu_msm(fq_coeffs, &bases),
//!     || cpu_msm(coeffs, bases),
//! );
//!
//! // Option 2: Manual type check with callback
//! if is_g1_affine::<C>() {
//!     with_fq_slice(coeffs, |fq| do_gpu_stuff(fq));
//! }
//! ```

// =============================================================================
// Re-exports from midnight-bls12-381-cuda
// =============================================================================

/// The backend line this fork is built on: the typed surface of
/// `midnight-bls12-381-cuda` instantiated for this fork's `midnight_curves`
/// package. The line name below is the ONLY line-specific token in this crate;
/// everything typed is reached through `backend::…`, and the curves-free core
/// (`config`, `diag`, `is_gpu_available`, …) is the same through this path or
/// the crate root. `_backend_line_is_this_forks_curves` fails the build if the
/// line's curves types are not this fork's.
#[cfg(feature = "gpu")]
pub use midnight_bls12_381_cuda::v03 as backend;

#[cfg(feature = "gpu")]
pub use backend::{
    // Core types
    GpuMsmContext,
    PrecomputedBases,
    TypeConverter,

    // MSM handles
    msm::MsmHandle,
    BatchMsmHandle,

    // Backend initialization
    ensure_backend_loaded,

    // Threshold functions (re-exported as-is for GPU code paths)
    should_use_gpu as backend_should_use_gpu,
    should_use_gpu_batch,
    min_gpu_size,
    min_ntt_gpu_size,

    // Autotune calibration (GPU/CPU crossovers + MSM window, measured once per
    // machine and persisted). A server that reaches the backend only through
    // this crate mounts `autotune_cli` as a subcommand and reads the effective
    // values with `calibration_report`. `min_gpu_size` / `min_ntt_gpu_size`
    // return the crossovers this crate's prover routes by; unless pinned by
    // its env override, the first call to either loads the persisted
    // calibration into this backend instance (measuring if there is none).
    autotune_cli,
    run_calibration,
    calibration_report,
    CalibrationReport,
    Provenance,

    // Trait-based API
    global_accelerator,
    GpuAccelerator,
    GpuCachedBases,
    MsmBackend,
    NttBackend,

    // Type-safe dispatch helpers (the key abstraction)
    is_fq, is_g1_affine,
    try_as_fq_slice, try_as_fq_slice_mut,
    projective_to_curve,
    dispatch_ntt_inplace,
    with_fq_slice_mut,
};

/// Compile-time wiring guard: this crate builds only if the backend line named
/// above takes this fork's `midnight_curves` types, i.e. the line's curves
/// package and this fork's are one package (the workspace `[patch.crates-io]`
/// redirect). A fork wired to the other line fails here with a type mismatch.
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn _backend_line_is_this_forks_curves(
    f: midnight_curves::Fq,
    a: midnight_curves::G1Affine,
    p: midnight_curves::G1Projective,
) -> (backend::curves::Fq, backend::curves::G1Affine, backend::curves::G1Projective) {
    (f, a, p)
}

/// View a generic `&[C]` as this fork's `&[midnight_curves::G1Affine]` iff `C`
/// is that type (an exact `TypeId` check through the backend line's
/// `is_g1_affine`, so a wrong-line type is reported, not reinterpreted).
#[cfg(feature = "gpu")]
#[inline]
pub fn try_as_g1_affine_slice<C: group::prime::PrimeCurveAffine + 'static>(
    slice: &[C],
) -> Option<&[G1Affine]> {
    if is_g1_affine::<C>() {
        // SAFETY: the TypeId check guarantees C == G1Affine, the same type.
        Some(unsafe { &*(slice as *const [C] as *const [G1Affine]) })
    } else {
        None
    }
}

// Re-export ICICLE runtime types needed by params.rs
#[cfg(feature = "gpu")]
pub use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
    Device as IcicleDevice,
    set_device as icicle_set_device,
};

#[cfg(feature = "gpu")]
use midnight_curves::{Fq, G1Affine, G1Projective};

// =============================================================================
// Public API (thin wrappers with alternate names to avoid collision)
// =============================================================================

/// Initialize GPU backend eagerly.
///
/// Call this at application startup to avoid first-request latency.
/// Returns the warmup duration if GPU is available.
#[cfg(feature = "gpu")]
pub fn init_gpu_backend() -> Option<std::time::Duration> {
    use tracing::info;
    
    let accel = global_accelerator();

    if let Err(e) = accel.initialize() {
        // GPU backend failed to load — the "backend-dir trap" (e.g. a wrong
        // ICICLE_BACKEND_INSTALL_DIR) that otherwise makes every op run on CPU
        // silently. Observable + strict-fatal so a misconfigured GPU deploy
        // fails loudly instead of degrading to CPU unnoticed.
        midnight_bls12_381_cuda::config::note_gpu_fallback("backend init", &e.to_string());
        return None;
    }

    match accel.warmup() {
        Ok(duration) => {
            info!("GPU backend warmed up in {:?}", duration);
            Some(duration)
        }
        Err(e) => {
            // Warmup is best-effort (a perf pre-load, not correctness): make it
            // visible but NOT strict-fatal — it does not imply later ops fail.
            eprintln!("[GPU-WARN] GPU warmup failed (non-fatal): {}", e);
            None
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub fn init_gpu_backend() -> Option<std::time::Duration> {
    None
}

/// Check if GPU acceleration is available.
#[cfg(feature = "gpu")]
pub fn is_gpu_available() -> bool {
    MsmBackend::is_gpu_available(global_accelerator())
}

#[cfg(not(feature = "gpu"))]
pub fn is_gpu_available() -> bool {
    false
}

/// Check if GPU should be used for the given size.
/// 
/// This is a wrapper that provides a non-GPU fallback and checks
/// both GPU availability and the size threshold.
#[cfg(feature = "gpu")]
pub fn should_use_gpu(size: usize) -> bool {
    backend_should_use_gpu(size)
}

#[cfg(not(feature = "gpu"))]
pub fn should_use_gpu(_size: usize) -> bool {
    false
}

/// Check if GPU should be used for NTT of the given size.
#[cfg(feature = "gpu")]
pub fn should_use_gpu_ntt(size: usize) -> bool {
    NttBackend::should_use_gpu(global_accelerator(), size)
}

#[cfg(not(feature = "gpu"))]
pub fn should_use_gpu_ntt(_size: usize) -> bool {
    false
}

// =============================================================================
// GPU Bases Handle (for SRS caching)
// =============================================================================

/// Handle to GPU-cached bases.
///
/// This type wraps the backend's cached bases and can be stored in
/// proof parameters for efficient MSM operations.
#[cfg(feature = "gpu")]
pub struct GpuBasesHandle {
    inner: Box<dyn GpuCachedBases>,
}

#[cfg(feature = "gpu")]
impl GpuBasesHandle {
    /// Create from backend's cached bases.
    pub fn new<T: GpuCachedBases + 'static>(bases: T) -> Self {
        Self {
            inner: Box::new(bases),
        }
    }
    
    /// Get the number of bases.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for GpuBasesHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBasesHandle")
            .field("len", &self.inner.len())
            .field("precompute_factor", &self.inner.precompute_factor())
            .finish()
    }
}

// =============================================================================
// MSM Operations
// =============================================================================

/// Compute MSM with GPU acceleration (if available and beneficial).
///
/// This function automatically chooses between GPU and CPU based on size
/// and availability.
#[cfg(feature = "gpu")]
pub fn msm_auto(scalars: &[Fq], bases: &[G1Affine]) -> G1Projective {
    let accel = global_accelerator();
    
    if MsmBackend::should_use_gpu(accel, scalars.len()) {
        match accel.msm(scalars, bases) {
            Ok(result) => return result,
            Err(e) => {
                midnight_bls12_381_cuda::config::note_gpu_fallback("MSM (msm_auto)", &e.to_string());
            }
        }
    }

    // CPU fallback via BLST
    msm_blst(scalars, bases)
}

/// Compute MSM with BLST (CPU).
#[cfg(feature = "gpu")]
fn msm_blst(scalars: &[Fq], bases: &[G1Affine]) -> G1Projective {
    let proj_bases: Vec<G1Projective> = bases.iter().map(|a| (*a).into()).collect();
    G1Projective::multi_exp(&proj_bases, scalars)
}

// =============================================================================
// SRS Base Caching
// =============================================================================

/// Upload SRS bases to GPU memory for efficient MSM.
///
/// Returns a handle that can be stored in proof parameters.
#[cfg(feature = "gpu")]
pub fn upload_bases(bases: &[G1Affine]) -> Result<GpuBasesHandle, String> {
    let accel = global_accelerator();
    
    accel
        .upload_bases(bases)
        .map(|cached| GpuBasesHandle::new(cached))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_availability() {
        // This should work regardless of GPU availability
        let available = is_gpu_available();
        println!("GPU available: {}", available);
    }
    
    #[test]
    fn test_should_use_gpu_thresholds() {
        // Small sizes should not use GPU
        assert!(!should_use_gpu(100));

        // Check NTT threshold
        assert!(!should_use_gpu_ntt(100));
    }

    /// Runtime twin of `_backend_line_is_this_forks_curves`: the backend
    /// line's dispatch recognises this fork's curves types by `TypeId`.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_backend_line_is_this_forks_curves() {
        assert!(is_fq::<midnight_curves::Fq>());
        assert!(is_g1_affine::<midnight_curves::G1Affine>());
        assert!(try_as_g1_affine_slice::<midnight_curves::G1Affine>(&[]).is_some());
    }
}
