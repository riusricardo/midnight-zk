//! Shared by the test binaries (`mod common;`).
//!
//! With `gpu`: the GPU precondition, and the check that the GPU did the work a
//! test claims for it. The GPU-op counters, the fallback counter, the device
//! mode and the calibration are process-wide, and the harness runs a binary's
//! tests in parallel. Every GPU test therefore takes `gpu_test()` first: it
//! serializes them, so a counter window sees only its own test's work.
//!
//! In every build: the verifier's pairing check computed with BLST alone, and a
//! tamper whose bytes still parse, so a rejection comes from the pairing.
#![allow(dead_code)] // each test binary uses part of it

use group::{Curve, Group, GroupEncoding};
use midnight_curves::{
    pairing::{MillerLoopResult, MultiMillerLoop},
    Bls12, Fq, G1Affine, G1Projective, G2Prepared,
};
use midnight_proofs::{
    plonk::Error,
    poly::kzg::{msm::DualMSM, params::ParamsKZG},
};

/// The pairing check of a prepared proof (the equation `DualMSM::check`
/// evaluates) with BLST alone: the guard's two MSMs by
/// `G1Projective::multi_exp`. `guard.verify` computes them through
/// `msm_specific`, which routes by size: on the GPU under forced GPU mode.
pub fn blst_pairing_check(params: &ParamsKZG<Bls12>, guard: &DualMSM<Bls12>) -> bool {
    let msm = |terms: Vec<(&Fq, &G1Projective)>| {
        let (scalars, bases): (Vec<Fq>, Vec<G1Projective>) =
            terms.into_iter().map(|(s, b)| (*s, *b)).unzip();
        G1Projective::multi_exp(&bases, &scalars).to_affine()
    };
    let (left, right) = guard.split();
    let (left, right) = (msm(left), msm(right));
    let s_g2 = G2Prepared::from(params.s_g2().to_affine());
    let n_g2 = G2Prepared::from((-params.g2()).to_affine());
    Bls12::multi_miller_loop(&[(&left, &s_g2), (&right, &n_g2)])
        .final_exponentiation()
        .is_identity()
        .into()
}

/// What the BLST check makes of a proof: `prepare` (decoding and the
/// transcript) fails, or the pairing rejects or accepts it.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Unparsable,
    Rejected,
    Accepted,
}

impl Verdict {
    /// The verdict on `prepare`'s result for a proof.
    pub fn of(params: &ParamsKZG<Bls12>, prepared: Result<DualMSM<Bls12>, Error>) -> Self {
        match prepared {
            Err(_) => Verdict::Unparsable,
            Ok(guard) if blst_pairing_check(params, &guard) => Verdict::Accepted,
            Ok(_) => Verdict::Rejected,
        }
    }
}

/// `proof` with its first commitment, the compressed G1 point it starts with,
/// moved to another valid point (P + G). The bytes still parse, so a verifier
/// must reject the proof at the pairing, not at decoding (a flipped bit of a
/// compressed point most likely fails decoding). Panics if the proof does not
/// start with a point.
pub fn tamper_first_commitment(proof: &[u8]) -> Vec<u8> {
    let mut repr = <G1Affine as GroupEncoding>::Repr::default();
    let len = repr.as_ref().len();
    repr.as_mut().copy_from_slice(&proof[..len]);
    let point: G1Affine =
        Option::from(G1Affine::from_bytes(&repr)).expect("the proof does not start with a G1 point");
    let moved = (G1Projective::from(point) + G1Projective::generator()).to_affine();
    let mut tampered = proof.to_vec();
    tampered[..len].copy_from_slice(moved.to_bytes().as_ref());
    tampered
}

#[cfg(feature = "gpu")]
pub use gpu::*;

#[cfg(feature = "gpu")]
mod gpu {
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
}
