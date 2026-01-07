//! Async vs Sync MSM Performance Comparison
//!
//! This benchmark tests the performance improvement from async GPU operations.
//!
//! Run with:
//!   cargo test --test async_msm_benchmark --features gpu --release -- --ignored --nocapture

#![cfg(feature = "gpu")]

use ff::Field;
use group::{Curve, Group};
use midnight_curves::{Fq as Scalar, G1Affine, G1Projective};
use midnight_bls12_381_cuda::{GpuMsmContext, TypeConverter};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

#[test]
#[ignore]
fn async_vs_sync_msm_benchmark() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              Async vs Sync MSM Performance Comparison                         ║");
    println!("║  Measuring the impact of removing synchronization overhead                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU context
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    println!("GPU Context initialized\n");

    // Test sizes (K values)
    let test_sizes = vec![14, 15, 16, 17, 18];

    println!("┌─────┬────────────┬──────────────┬──────────────┬──────────────┬─────────────┐");
    println!("│  K  │   Points   │  Sync (ms)   │  Async (ms)  │  Speedup     │  Status     │");
    println!("├─────┼────────────┼──────────────┼──────────────┼──────────────┼─────────────┤");

    for k in test_sizes {
        let size = 1 << k;
        let mut rng = ChaCha8Rng::seed_from_u64(k as u64);

        // Generate test data
        let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
        let points_proj: Vec<G1Projective> = (0..size)
            .map(|_| G1Projective::random(&mut rng))
            .collect();
        let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();

        // Upload bases to GPU (one-time cost)
        let device_bases = ctx
            .upload_g1_bases(&points)
            .expect("Failed to upload bases");

        // Warmup
        let _ = ctx
            .msm_with_device_bases(&scalars, &device_bases)
            .expect("Warmup failed");

        // Benchmark SYNC mode (current implementation)
        let sync_start = Instant::now();
        let sync_result = ctx
            .msm_with_device_bases(&scalars, &device_bases)
            .expect("Sync MSM failed");
        let sync_time = sync_start.elapsed().as_secs_f64() * 1000.0;

        // Benchmark ASYNC mode (new implementation)
        let async_start = Instant::now();
        let async_handle = ctx
            .msm_with_device_bases_async(&scalars, &device_bases)
            .expect("Async MSM launch failed");

        // Simulate CPU work while GPU computes (in real usage, you'd do actual work here)
        // For benchmark, just measure the overhead difference

        let async_result = async_handle
            .wait()
            .expect("Async MSM wait failed");
        let async_time = async_start.elapsed().as_secs_f64() * 1000.0;

        // Verify results match
        assert_eq!(
            sync_result, async_result,
            "Sync and async results must match!"
        );

        let speedup = sync_time / async_time;
        let status = if speedup > 1.5 {
            "✓ FASTER"
        } else if speedup > 0.95 {
            "≈ EQUAL"
        } else {
            "✗ SLOWER"
        };

        println!(
            "│ {:>3} │ {:>10} │ {:>12.2} │ {:>12.2} │ {:>12.2}x │ {:>11} │",
            k, size, sync_time, async_time, speedup, status
        );
    }

    println!("└─────┴────────────┴──────────────┴──────────────┴──────────────┴─────────────┘");
    println!();
    println!("Interpretation:");
    println!("  • Speedup > 1.5x: Async mode provides significant improvement");
    println!("  • Speedup ≈ 1.0x: Kernel execution dominates (expected for very large K)");
    println!("  • Speedup < 1.0x: Async overhead (should not happen - investigate)");
    println!();
    println!("Expected results:");
    println!("  • K=13-15: 2-4x speedup (sync overhead is significant)");
    println!("  • K=16-17: 1.5-2x speedup (kernel time dominates)");
    println!("  • K≥18: ~1.2x speedup (limited by kernel execution)");
}

#[test]
#[ignore]
fn async_msm_correctness() {
    println!("Testing async MSM correctness...");

    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let size = 1 << 16;
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let points_proj: Vec<G1Projective> = (0..size)
        .map(|_| G1Projective::random(&mut rng))
        .collect();
    let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();

    // Upload bases
    let device_bases = ctx
        .upload_g1_bases(&points)
        .expect("Failed to upload bases");

    // Compute with CPU
    let cpu_result = G1Projective::multi_exp(&points_proj, &scalars);

    // Compute with GPU sync
    let gpu_sync_result = ctx
        .msm_with_device_bases(&scalars, &device_bases)
        .expect("Sync MSM failed");

    // Compute with GPU async
    let async_handle = ctx
        .msm_with_device_bases_async(&scalars, &device_bases)
        .expect("Async MSM launch failed");
    let gpu_async_result = async_handle
        .wait()
        .expect("Async MSM wait failed");

    // Verify all match
    assert_eq!(cpu_result, gpu_sync_result, "CPU and GPU sync results must match");
    assert_eq!(cpu_result, gpu_async_result, "CPU and GPU async results must match");
    assert_eq!(gpu_sync_result, gpu_async_result, "GPU sync and async results must match");

    println!("✓ All results match - async implementation is correct");
}

#[test]
#[ignore]
fn async_msm_pipeline_test() {
    println!("Testing async MSM pipelining capability...");

    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    let size = 1 << 15;
    let num_batches = 3;

    // Create multiple batches
    let mut batches = Vec::new();
    for i in 0..num_batches {
        let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
        let points_proj: Vec<G1Projective> = (0..size)
            .map(|_| G1Projective::random(&mut rng))
            .collect();
        let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();

        let device_bases = ctx.upload_g1_bases(&points).expect("Upload failed");

        batches.push((scalars, points_proj, device_bases));
    }

    // Sequential (sync) execution
    let seq_start = Instant::now();
    let mut seq_results = Vec::new();
    for (scalars, _, device_bases) in &batches {
        let result = ctx
            .msm_with_device_bases(scalars, device_bases)
            .expect("Sync MSM failed");
        seq_results.push(result);
    }
    let seq_time = seq_start.elapsed();

    // Pipelined (async) execution - launch all, then wait
    let pipe_start = Instant::now();
    let mut handles = Vec::new();
    for (scalars, _, device_bases) in &batches {
        let handle = ctx
            .msm_with_device_bases_async(scalars, device_bases)
            .expect("Async launch failed");
        handles.push(handle);
    }

    // Now wait for all results
    let mut pipe_results = Vec::new();
    for handle in handles {
        let result = handle.wait().expect("Wait failed");
        pipe_results.push(result);
    }
    let pipe_time = pipe_start.elapsed();

    // Verify results match
    assert_eq!(seq_results, pipe_results, "Sequential and pipelined results must match");

    let speedup = seq_time.as_secs_f64() / pipe_time.as_secs_f64();

    println!("Sequential time:  {:.2}ms", seq_time.as_secs_f64() * 1000.0);
    println!("Pipelined time:   {:.2}ms", pipe_time.as_secs_f64() * 1000.0);
    println!("Speedup:          {:.2}x", speedup);
    println!();

    if speedup > 1.2 {
        println!("✓ Pipelining provides {:.2}x speedup", speedup);
    } else {
        println!("⚠ Limited pipelining benefit - kernel execution may dominate");
    }
}
