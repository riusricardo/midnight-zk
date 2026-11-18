//! Shared by the GPU test binaries (`mod common;`): the GPU precondition, and the
//! check that the GPU did the work a test claims for it.
//!
//! The GPU-op counters, the fallback counter, the device mode and the
//! calibration are process-wide, and the harness runs a binary's tests in
//! parallel. Every GPU test therefore takes `gpu_test()` first: it serializes
//! them, so a counter window sees only its own test's work.
#![allow(dead_code)] // each test binary uses part of it

use std::io::Write;
use std::sync::{Mutex, MutexGuard};

use midnight_bls12_381_cuda::config::{gpu_fallback_count, gpu_strict};
use midnight_proofs::gpu_accel::backend::{
    self,
    diag::{gpu_op_count, OpKind},
};

static SERIAL: Mutex<()> = Mutex::new(());

/// Taken first by every GPU test. `None` means skip: there is no GPU backend
/// and `MIDNIGHT_GPU_STRICT=1` is not set (under strict mode, how every gate
/// runs, a missing GPU fails the test). The skip is written straight to stderr,
/// past the harness's output capture, so a run that skipped says so.
/// Otherwise a guard that keeps the binary's other GPU tests out of this one,
/// with the lazy calibration resolved (on a cold cache it measures, which would
/// move the counters inside a window).
pub fn gpu_test() -> Option<MutexGuard<'static, ()>> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let test = std::thread::current().name().unwrap_or("a GPU test").to_owned();
    if !midnight_bls12_381_cuda::is_gpu_available() {
        assert!(
            !gpu_strict(),
            "{test}: no GPU backend under MIDNIGHT_GPU_STRICT=1 (ICICLE_BACKEND_INSTALL_DIR?)"
        );
        let _ = writeln!(
            std::io::stderr(),
            "SKIPPED {test}: no GPU backend (ICICLE_BACKEND_INSTALL_DIR?); it checked nothing"
        );
        return None;
    }
    midnight_bls12_381_cuda::settle_calibration();
    Some(guard)
}

/// The log2 size, at least `k`, from which the routing sends an NTT to the GPU
/// in Auto mode as well as in forced GPU mode.
pub fn gpu_ntt_k(k: u32) -> u32 {
    k.max(midnight_bls12_381_cuda::min_ntt_gpu_size().trailing_zeros())
}

/// The log2 size, at least `k`, from which the routing sends an MSM to the GPU
/// in Auto mode as well as in forced GPU mode.
pub fn gpu_msm_k(k: u32) -> u32 {
    k.max(midnight_bls12_381_cuda::min_gpu_size().trailing_zeros())
}

/// Runs `f`, which must do its work on the GPU through this fork's backend
/// line: at least one launch of each kind in `kinds`, and no GPU->CPU fallback
/// anywhere in the process (fatal under strict mode; this catches one without
/// it). Call it under `gpu_test()`'s guard.
pub fn expect_gpu_ops<T>(kinds: &[OpKind], f: impl FnOnce() -> T) -> T {
    let line = backend::line_id();
    let launches = || kinds.iter().map(|&kind| gpu_op_count(line, kind)).collect::<Vec<_>>();
    let (before, fallbacks) = (launches(), gpu_fallback_count());
    let out = f();
    for ((kind, before), after) in kinds.iter().zip(before).zip(launches()) {
        assert!(after > before, "no GPU {kind:?} launched on backend line {line}");
    }
    assert_eq!(gpu_fallback_count(), fallbacks, "GPU work fell back to the CPU");
    out
}
