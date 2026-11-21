//! GPU integration tests
//!
//! These tests verify GPU acceleration with ICICLE backend.
//! Requires GPU feature and ICICLE backend installed.

#![cfg(feature = "gpu")]

mod common;

use common::{expect_gpu_ops, gpu_ntt_k, gpu_test, omega_for};
use midnight_proofs::gpu_accel::{
    backend::{self, diag::OpKind},
    GpuMsmContext,
};
use midnight_curves::{Fq as Scalar, G1Affine, G1Projective};
use ff::Field;
use group::{Curve, Group, prime::PrimeCurveAffine};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn test_gpu_msm_small() {
    let Some(_gpu) = gpu_test() else { return };
    // Small MSM: 1024 points
    let size = 1024;
    test_msm_with_size(size);
}

#[test]
fn test_gpu_msm_k14() {
    let Some(_gpu) = gpu_test() else { return };
    // K=14: 16384 points
    let size = 16384;
    test_msm_with_size(size);
}

#[test]
fn test_gpu_msm_k16() {
    let Some(_gpu) = gpu_test() else { return };
    // K=16: 65536 points
    let size = 65536;
    test_msm_with_size(size);
}

fn test_msm_with_size(size: usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(size as u64);

    // Generate random scalars and points
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::random(&mut rng).to_affine())
        .collect();

    // Create GPU MSM context
    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");

    // Execute MSM. The launch is counted on this fork's backend line, so a fork
    // wired to the other line's counters is caught at runtime as well as at
    // compile time.
    let result = expect_gpu_ops(&[OpKind::Msm], || ctx.msm(&scalars, &points)).expect("MSM should succeed");

    // Verify against CPU reference
    let points_proj: Vec<G1Projective> = points.iter().map(|p| G1Projective::from(*p)).collect();
    let cpu_result = G1Projective::multi_exp(&points_proj, &scalars);
    assert_eq!(result, cpu_result, "GPU and CPU results should match");
}

/// The prover's NTT entry point (`forward_ntt_inplace_auto`, which
/// `utils::fft::best_fft` routes to) launches on this fork's backend line, at a
/// size the routing sends to the GPU, and gives `best_fft`'s result.
#[test]
fn test_gpu_ntt_counted_on_this_forks_line() {
    let Some(_gpu) = gpu_test() else { return };
    let k = gpu_ntt_k(14);
    let n = 1usize << k;
    let mut data: Vec<Scalar> = (0..n).map(|i| Scalar::from(i as u64 + 1)).collect();
    let mut want = data.clone();
    midnight_curves::fft::best_fft(&mut want, omega_for(k), k);
    expect_gpu_ops(&[OpKind::Ntt], || backend::ntt::forward_ntt_inplace_auto(&mut data))
        .expect("NTT should succeed");
    assert!(data == want, "the NTT of 2^{k} elements differs from best_fft");
}

/// The start-up entry (`init_gpu_backend`, which the ledger calls before a
/// proof server serves) loads the backend, warms the GPU up with an MSM on this
/// fork's line and reports the warm-up.
#[test]
fn test_init_gpu_backend_warms_the_gpu_up() {
    let Some(_gpu) = gpu_test() else { return };
    let warmup = expect_gpu_ops(&[OpKind::Msm], midnight_proofs::init_gpu_backend);
    assert!(warmup.is_some(), "init_gpu_backend reported no warm-up");
}

#[test]
fn test_gpu_msm_simple_identity() {
    let Some(_gpu) = gpu_test() else { return };
    // Simple test: MSM with scalar=1 and generator should return n * G
    let generator = G1Affine::generator();
    let n = 1024;
    let scalars = vec![Scalar::ONE; n];
    let points = vec![generator; n];

    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");

    // This computes sum of n generators = n * G
    let result = expect_gpu_ops(&[OpKind::Msm], || ctx.msm(&scalars, &points)).expect("MSM should succeed");

    // Expected: n * G
    let expected = G1Projective::from(generator) * Scalar::from(n as u64);
    assert_eq!(result, expected, "MSM result should be n * G");
}
