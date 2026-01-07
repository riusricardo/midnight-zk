//! Integration test for batch MSM functionality
//!
//! This validates that the Phase 1 batch MSM optimization works correctly
//! end-to-end through the proofs crate API.

#![cfg(feature = "gpu")]

use ff::Field;
use group::prime::PrimeCurveAffine;
use midnight_curves::{Fq as Scalar, G1Affine, G1Projective};
use midnight_proofs::poly::kzg::msm::{msm_batch_with_cached_bases, msm_with_cached_bases, init_gpu_backend};
use midnight_bls12_381_cuda::GpuMsmContext;

#[test]
fn test_batch_msm_correctness() {
    // Initialize GPU
    let _ = init_gpu_backend();
    
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    
    // Test parameters
    let batch_size = 8;
    let msm_size = 1024; // 2^10
    
    // Create test data: different scalars for each MSM
    let scalars_batch: Vec<Vec<Scalar>> = (0..batch_size)
        .map(|batch_idx| {
            (0..msm_size)
                .map(|i| Scalar::from((batch_idx * 1000 + i + 1) as u64))
                .collect()
        })
        .collect();
    
    // Shared bases (simulating SRS)
    let bases: Vec<G1Affine> = (0..msm_size)
        .map(|_| G1Affine::generator())
        .collect();
    
    // Upload bases to GPU once
    let device_bases = ctx.upload_g1_bases(&bases).expect("Upload failed");
    
    // Test 1: Batch MSM produces correct results
    println!("Test 1: Batch MSM correctness");
    let scalar_refs: Vec<&[Scalar]> = scalars_batch.iter().map(|v| &v[..]).collect();
    
    let batch_results: Vec<G1Projective> = msm_batch_with_cached_bases::<G1Affine>(&scalar_refs, &device_bases);
    
    assert_eq!(batch_results.len(), batch_size, "Wrong number of results");
    
    // Verify each result matches individual MSM
    for (i, scalars) in scalars_batch.iter().enumerate() {
        let individual_result: G1Projective = msm_with_cached_bases::<G1Affine>(scalars, &device_bases);
        assert_eq!(
            batch_results[i],
            individual_result,
            "Batch result {} doesn't match individual MSM",
            i
        );
    }
    
    println!("✓ All {} batch results match individual MSMs", batch_size);
}

#[test]
fn test_batch_msm_empty() {
    let _ = init_gpu_backend();
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    
    let bases = vec![G1Affine::generator()];
    let device_bases = ctx.upload_g1_bases(&bases).unwrap();
    
    let empty_batch: Vec<&[Scalar]> = vec![];
    let results: Vec<G1Projective> = msm_batch_with_cached_bases::<G1Affine>(&empty_batch, &device_bases);
    
    assert_eq!(results.len(), 0);
    println!("✓ Empty batch handled correctly");
}

#[test]
fn test_batch_msm_single() {
    let _ = init_gpu_backend();
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    
    let msm_size = 256;
    let scalars = vec![Scalar::from(42u64); msm_size];
    let bases = vec![G1Affine::generator(); msm_size];
    
    let device_bases = ctx.upload_g1_bases(&bases).unwrap();
    
    // Batch of size 1
    let batch = vec![&scalars[..]];
    let batch_results: Vec<G1Projective> = msm_batch_with_cached_bases::<G1Affine>(&batch, &device_bases);
    
    let single_result: G1Projective = msm_with_cached_bases::<G1Affine>(&scalars, &device_bases);
    
    assert_eq!(batch_results.len(), 1);
    assert_eq!(batch_results[0], single_result);
    println!("✓ Single-element batch works correctly");
}

#[test]
fn test_batch_msm_different_sizes() {
    let _ = init_gpu_backend();
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    
    let sizes = [512, 1024, 2048];
    
    for &size in &sizes {
        let scalars: Vec<Vec<Scalar>> = (0..4)
            .map(|_| (0..size).map(|i| Scalar::from((i + 1) as u64)).collect())
            .collect();
        
        let bases = vec![G1Affine::generator(); size];
        let device_bases = ctx.upload_g1_bases(&bases).unwrap();
        
        let scalar_refs: Vec<&[Scalar]> = scalars.iter().map(|v| &v[..]).collect();
        let results: Vec<G1Projective> = msm_batch_with_cached_bases::<G1Affine>(&scalar_refs, &device_bases);
        
        assert_eq!(results.len(), 4);
        println!("✓ Batch MSM works for size {}", size);
    }
}

#[test]
#[should_panic(expected = "same size")]
fn test_batch_msm_size_mismatch_panics() {
    let _ = init_gpu_backend();
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    
    let scalars1 = vec![Scalar::ONE; 100];
    let scalars2 = vec![Scalar::ONE; 200]; // Different size!
    
    let bases = vec![G1Affine::generator(); 200];
    let device_bases = ctx.upload_g1_bases(&bases).unwrap();
    
    let batch = vec![&scalars1[..], &scalars2[..]];
    let _: Vec<G1Projective> = msm_batch_with_cached_bases::<G1Affine>(&batch, &device_bases);
}

#[test]
fn test_batch_msm_performance_indicator() {
    use std::time::Instant;
    
    let _ = init_gpu_backend();
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    
    let batch_size = 8;
    let msm_size = 2048; // 2^11
    
    let scalars_batch: Vec<Vec<Scalar>> = (0..batch_size)
        .map(|i| vec![Scalar::from((i + 1) as u64); msm_size])
        .collect();
    
    let bases = vec![G1Affine::generator(); msm_size];
    let device_bases = ctx.upload_g1_bases(&bases).unwrap();
    
    // Warmup
    let scalar_refs: Vec<&[Scalar]> = scalars_batch.iter().map(|v| &v[..]).collect();
    let _: Vec<G1Projective> = msm_batch_with_cached_bases::<G1Affine>(&scalar_refs, &device_bases);
    
    // Benchmark sequential
    let start = Instant::now();
    for scalars in &scalars_batch {
        let _: G1Projective = msm_with_cached_bases::<G1Affine>(scalars, &device_bases);
    }
    let sequential_time = start.elapsed();
    
    // Benchmark batched
    let start = Instant::now();
    let _: Vec<G1Projective> = msm_batch_with_cached_bases::<G1Affine>(&scalar_refs, &device_bases);
    let batch_time = start.elapsed();
    
    let speedup = sequential_time.as_secs_f64() / batch_time.as_secs_f64();
    
    println!("\n=== Performance Comparison ===");
    println!("Sequential ({} MSMs): {:?}", batch_size, sequential_time);
    println!("Batched (single call): {:?}", batch_time);
    println!("Speedup: {:.2}x", speedup);
    println!("==============================\n");
    
    // We expect at least some speedup (conservative check)
    // In practice should see 4-8x on real hardware
    assert!(
        speedup > 1.0,
        "Batch MSM should be faster than sequential (got {:.2}x)",
        speedup
    );
}
