/*
 * Copyright (C) 2026 Midnight Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! The GPU bridge: everything this crate takes from the CUDA backend
//! (`midnight-bls12-381-cuda`) is re-exported here, from one backend line, so
//! the rest of the crate imports GPU items from `crate::gpu_accel` only.

// =============================================================================
// Re-exports from midnight-bls12-381-cuda
// =============================================================================

/// The backend line this fork is built on: the typed surface of
/// `midnight-bls12-381-cuda` instantiated for this fork's `midnight_curves`
/// package. The line name below is the ONLY line-specific token in this crate;
/// everything typed is reached through `backend::…`, and the curves-free core
/// (`config`, `diag`, `is_gpu_available`, …) is the same through this path or
/// the crate root. `_BACKEND_LINE_IS_THIS_FORKS_CURVES` fails the build if the
/// line's curves types are not this fork's.
pub use midnight_bls12_381_cuda::v02 as backend;

pub use backend::{
    // MSM context, its async handle, and SRS bases on the device
    GpuMsmContext,
    msm::MsmHandle,
    PrecomputedBases,

    // Routing (`MIDNIGHT_DEVICE`, the calibrated crossovers, whether the GPU
    // can be used) and the one-line description of the backend
    should_use_gpu,
    should_use_gpu_ntt,
    min_gpu_size,
    min_ntt_gpu_size,
    backend_info,

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

    // Type-checked views of this crate's generic types as the line's
    is_fq, is_g1_affine,
    try_as_fq_slice, try_as_fq_slice_mut, try_as_g1_affine_slice,
    projective_to_curve,
    dispatch_ntt_inplace,
};

/// Compile-time wiring guard: this crate builds only if the backend line named
/// above takes this fork's `midnight_curves` types, i.e. the line's curves
/// package and this fork's are one package (the workspace `[patch.crates-io]`
/// redirect). A fork wired to the other line fails here with a type mismatch.
const _BACKEND_LINE_IS_THIS_FORKS_CURVES: fn(
    midnight_curves::Fq,
    midnight_curves::G1Affine,
    midnight_curves::G1Projective,
) -> (backend::curves::Fq, backend::curves::G1Affine, backend::curves::G1Projective) = |f, a, p| (f, a, p);


/// Loads the GPU backend, selects the device and warms the GPU up, at start-up,
/// so that the first proof does not pay for it. Returns the warm-up time, or
/// `None` when the GPU cannot be used (a missing backend in
/// `ICICLE_BACKEND_INSTALL_DIR`, a device that does not exist: reported through
/// `note_gpu_fallback`, so a misconfigured GPU deployment fails under
/// `MIDNIGHT_GPU_STRICT=1` instead of running on the CPU unnoticed) or when the
/// warm-up failed (reported, not fatal: it is a pre-load, not correctness).
pub fn init_gpu_backend() -> Option<std::time::Duration> {
    if let Err(e) = backend::initialize() {
        midnight_bls12_381_cuda::config::note_gpu_fallback("backend init", &e);
        return None;
    }
    match GpuMsmContext::new().and_then(|ctx| ctx.warmup()) {
        Ok(duration) => {
            tracing::info!("GPU backend warmed up in {:?}", duration);
            Some(duration)
        }
        Err(e) => {
            eprintln!("[GPU-WARN] GPU warmup failed (non-fatal): {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runtime twin of `_BACKEND_LINE_IS_THIS_FORKS_CURVES`: the backend
    /// line's dispatch recognises this fork's curves types by `TypeId`.
    #[test]
    fn test_backend_line_is_this_forks_curves() {
        assert!(is_fq::<midnight_curves::Fq>());
        assert!(is_g1_affine::<midnight_curves::G1Affine>());
        assert!(try_as_g1_affine_slice::<midnight_curves::G1Affine>(&[]).is_some());
    }
}
