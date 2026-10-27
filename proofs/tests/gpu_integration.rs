//! GPU integration tests
//!
//! These tests verify GPU acceleration with ICICLE backend.
//! Requires GPU feature and ICICLE backend installed.

#![cfg(feature = "gpu")]

use midnight_proofs::gpu_accel::{
    backend::{
        self,
        diag::{gpu_op_count, OpKind},
    },
    GpuMsmContext,
};
use midnight_curves::{Fq as Scalar, G1Affine, G1Projective};
use ff::Field;
use group::{Group, Curve, prime::PrimeCurveAffine};

#[test]
fn test_gpu_msm_small() {
    // Small MSM: 1024 points
    let size = 1024;
    test_msm_with_size(size);
}

#[test]
fn test_gpu_msm_k14() {
    // K=14: 16384 points
    let size = 16384;
    test_msm_with_size(size);
}

#[test]
fn test_gpu_msm_k16() {
    // K=16: 65536 points
    let size = 65536;
    test_msm_with_size(size);
}

fn test_msm_with_size(size: usize) {
    let mut rng = rand::thread_rng();

    // Generate random scalars and points
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let points: Vec<G1Affine> = (0..size)
        .map(|_| G1Projective::random(&mut rng).to_affine())
        .collect();

    // Create GPU MSM context
    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");

    // Execute MSM
    let line = backend::line_id();
    let msm_launches_before = gpu_op_count(line, OpKind::Msm);
    let result = ctx.msm(&scalars, &points).expect("MSM should succeed");

    // The launch is counted on this fork's backend line (k >= 14: the sizes the
    // prover routes to the GPU), so a fork wired to the other line's counters
    // is caught at runtime as well as at compile time.
    if size >= 1 << 14 {
        assert!(
            gpu_op_count(line, OpKind::Msm) > msm_launches_before,
            "MSM of {size} points was not counted as a GPU launch on backend line {line}"
        );
    }

    // Verify result is not identity (with random inputs, extremely unlikely)
    assert!(!bool::from(result.is_identity()));

    // Verify against CPU reference
    let points_proj: Vec<G1Projective> = points.iter().map(|p| G1Projective::from(*p)).collect();
    let cpu_result = G1Projective::multi_exp(&points_proj, &scalars);
    assert_eq!(result, cpu_result, "GPU and CPU results should match");
}

/// The prover's NTT entry point (`forward_ntt_inplace_auto`, which
/// `utils::fft::best_fft` routes to) launches on this fork's backend line: its
/// NTT counter moves for a k = 14 transform whenever the device routing sends
/// that size to the GPU (`MIDNIGHT_DEVICE=cpu` keeps it on the CPU arm).
#[test]
fn test_gpu_ntt_counted_on_this_forks_line() {
    let line = backend::line_id();
    let n = 1usize << 14;
    let mut data: Vec<Scalar> = (0..n).map(|i| Scalar::from(i as u64 + 1)).collect();
    let ntt_launches_before = gpu_op_count(line, OpKind::Ntt);
    backend::ntt::forward_ntt_inplace_auto(&mut data).expect("NTT should succeed");
    if backend::should_use_gpu_ntt(n) {
        assert!(
            gpu_op_count(line, OpKind::Ntt) > ntt_launches_before,
            "NTT of {n} elements was not counted as a GPU launch on backend line {line}"
        );
    } else {
        println!("[skip] NTT of {n} elements is routed to the CPU; no GPU launch expected");
    }
}

#[test]
fn test_gpu_context_creation() {
    let ctx = GpuMsmContext::new();
    assert!(ctx.is_ok(), "GPU context should initialize successfully");
}

#[test]
fn test_gpu_msm_simple_identity() {
    // Simple test: MSM with scalar=1 and generator should return n * G
    let generator = G1Affine::generator();
    let n = 1024;
    let scalars = vec![Scalar::ONE; n];
    let points = vec![generator; n];

    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");

    // This computes sum of n generators = n * G
    let result = ctx.msm(&scalars, &points).expect("MSM should succeed");

    // Expected: n * G
    let expected = G1Projective::from(generator) * Scalar::from(n as u64);
    assert_eq!(result, expected, "MSM result should be n * G");
}

#[test]
fn test_gpu_msm_empty() {
    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");
    let result = ctx.msm(&[], &[]).expect("Empty MSM should succeed");
    assert!(bool::from(result.is_identity()), "Empty MSM should return identity");
}

// =============================================================================
// G2 MSM Tests
// =============================================================================

use midnight_curves::{G2Affine, G2Projective};

#[test]
#[ignore = "GPU G2 MSM is an intentional NOT_IMPLEMENTED stub (CPU pairings are used); un-ignore when G2 lands"]
fn test_gpu_g2_msm_small() {
    // Small G2 MSM: 1024 points
    let size = 1024;
    test_g2_msm_with_size(size);
}

#[test]
#[ignore = "GPU G2 MSM is an intentional NOT_IMPLEMENTED stub (CPU pairings are used); un-ignore when G2 lands"]
fn test_gpu_g2_msm_k14() {
    // K=14: 16384 points
    let size = 16384;
    test_g2_msm_with_size(size);
}

#[test]
#[ignore = "GPU G2 MSM is an intentional NOT_IMPLEMENTED stub (CPU pairings are used); un-ignore when G2 lands"]
fn test_gpu_g2_msm_k16() {
    // K=16: 65536 points
    let size = 65536;
    test_g2_msm_with_size(size);
}

fn test_g2_msm_with_size(size: usize) {
    let mut rng = rand::thread_rng();

    // Generate random scalars and G2 points
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let points: Vec<G2Affine> = (0..size)
        .map(|_| G2Projective::random(&mut rng).to_affine())
        .collect();

    // Create GPU MSM context
    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");

    // Execute G2 MSM
    let result = ctx.g2_msm(&scalars, &points).expect("G2 MSM should succeed");

    // Verify result is not identity (with random inputs, extremely unlikely)
    assert!(!bool::from(result.is_identity()), "G2 MSM result should not be identity");

    // Verify against CPU reference
    let points_proj: Vec<G2Projective> = points.iter().map(|p| G2Projective::from(*p)).collect();
    let cpu_result = G2Projective::multi_exp(&points_proj, &scalars);
    assert_eq!(result, cpu_result, "GPU and CPU G2 results should match");
}

#[test]
#[ignore = "GPU G2 MSM is an intentional NOT_IMPLEMENTED stub (CPU pairings are used); un-ignore when G2 lands"]
fn test_gpu_g2_msm_simple_identity() {
    // Simple test: MSM with scalar=1 and generator should return n * G2
    let generator = G2Affine::generator();
    let n = 1024;
    let scalars = vec![Scalar::ONE; n];
    let points = vec![generator; n];

    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");

    // This computes sum of n G2 generators = n * G2
    let result = ctx.g2_msm(&scalars, &points).expect("G2 MSM should succeed");

    // Expected: n * G2
    let expected = G2Projective::from(generator) * Scalar::from(n as u64);
    assert_eq!(result, expected, "G2 MSM result should be n * G2");
}

#[test]
#[ignore = "GPU G2 MSM is an intentional NOT_IMPLEMENTED stub (CPU pairings are used); un-ignore when G2 lands"]
fn test_gpu_g2_msm_zero_scalars() {
    // Test with all zero scalars - should return identity
    let size = 1024;
    let scalars = vec![Scalar::ZERO; size];
    let points: Vec<G2Affine> = (0..size)
        .map(|_| G2Projective::random(&mut rand::thread_rng()).to_affine())
        .collect();

    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");
    let result = ctx.g2_msm(&scalars, &points).expect("G2 MSM should succeed");

    assert!(
        bool::from(result.is_identity()),
        "G2 MSM with zero scalars should be identity"
    );
}

#[test]
fn test_gpu_warmup() {
    let ctx = GpuMsmContext::new().expect("Failed to create MSM context");
    let warmup_time = ctx.warmup();
    assert!(warmup_time.is_ok(), "Warmup should succeed");
    println!("GPU warmup completed in {:?}", warmup_time.unwrap());
}
