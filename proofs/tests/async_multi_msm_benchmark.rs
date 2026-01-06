//! Benchmark demonstrating the TRUE benefit of async MSM
//! 
//! This test shows realistic usage: computing multiple MSMs where
//! async mode enables pipelining and CPU/GPU overlap.

#![cfg(all(test, feature = "gpu"))]

use midnight_curves::{Fq as Scalar, G1Affine, G1Projective};
use midnight_proofs::gpu::GpuMsmContext;
use group::{Group, Curve};
use ff::Field;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use std::time::Instant;

const K: usize = 15; // 32K points - GPU threshold
const SIZE: usize = 1 << K;
const NUM_MSMS: usize = 10; // Multiple MSMs to show pipelining benefit

#[test]
#[ignore] // Run with: cargo test --test async_multi_msm_benchmark --features gpu --release -- --ignored --nocapture
fn benchmark_multiple_msms_sync_vs_async() {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Multi-MSM Benchmark: Sync vs Async Pipelining                       ║");
    println!("║  Computing {} MSMs with {} points each (K={})                        ║", NUM_MSMS, SIZE, K);
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate test data for multiple MSMs
    let mut all_scalars = Vec::new();
    let mut all_points = Vec::new();
    let mut device_bases = Vec::new();

    println!("Preparing {} batches of {} points...", NUM_MSMS, SIZE);
    for _ in 0..NUM_MSMS {
        let scalars: Vec<Scalar> = (0..SIZE).map(|_| Scalar::random(&mut rng)).collect();
        let points_proj: Vec<G1Projective> = (0..SIZE)
            .map(|_| G1Projective::random(&mut rng))
            .collect();
        let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();

        let bases = ctx.upload_g1_bases(&points).expect("Upload failed");

        all_scalars.push(scalars);
        all_points.push(points);
        device_bases.push(bases);
    }

    println!("✓ Data prepared\n");

    // =========================================================================
    // SYNC MODE: Process MSMs sequentially
    // =========================================================================
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("SYNC MODE: Sequential execution (wait after each MSM)");
    println!("─────────────────────────────────────────────────────────────────────────────");

    let sync_start = Instant::now();
    let mut sync_results = Vec::new();

    for i in 0..NUM_MSMS {
        let result = ctx
            .msm_with_device_bases(&all_scalars[i], &device_bases[i])
            .expect("Sync MSM failed");
        sync_results.push(result);
    }

    let sync_time = sync_start.elapsed();
    println!("  Total time: {:?}", sync_time);
    println!("  Per MSM:    {:?}", sync_time / NUM_MSMS as u32);
    println!();

    // =========================================================================
    // ASYNC MODE: Launch all MSMs, then wait
    // =========================================================================
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("ASYNC MODE: Pipeline execution (launch all, then wait all)");
    println!("─────────────────────────────────────────────────────────────────────────────");

    let async_start = Instant::now();
    let mut handles = Vec::new();

    // Launch all MSMs without waiting (GPU pipelining)
    let launch_start = Instant::now();
    for i in 0..NUM_MSMS {
        let handle = ctx
            .msm_with_device_bases_async(&all_scalars[i], &device_bases[i])
            .expect("Async launch failed");
        handles.push(handle);
    }
    let launch_time = launch_start.elapsed();

    // Now wait for all results
    let wait_start = Instant::now();
    let mut async_results = Vec::new();
    for handle in handles {
        let result = handle.wait().expect("Wait failed");
        async_results.push(result);
    }
    let wait_time = wait_start.elapsed();

    let async_time = async_start.elapsed();
    println!("  Launch time: {:?}", launch_time);
    println!("  Wait time:   {:?}", wait_time);
    println!("  Total time:  {:?}", async_time);
    println!("  Per MSM:     {:?}", async_time / NUM_MSMS as u32);
    println!();

    // Verify correctness
    assert_eq!(sync_results.len(), async_results.len());
    for (i, (sync_res, async_res)) in sync_results.iter().zip(async_results.iter()).enumerate() {
        assert_eq!(
            sync_res, async_res,
            "MSM {} results don't match!",
            i
        );
    }

    println!("✓ All results match - correctness verified");
    println!();

    // =========================================================================
    // Performance Analysis
    // =========================================================================
    let speedup = sync_time.as_secs_f64() / async_time.as_secs_f64();

    println!("═════════════════════════════════════════════════════════════════════════════");
    println!("PERFORMANCE RESULTS");
    println!("═════════════════════════════════════════════════════════════════════════════");
    println!("  Sync total:    {:?}", sync_time);
    println!("  Async total:   {:?}", async_time);
    println!("  Speedup:       {:.2}x", speedup);
    println!();

    if speedup > 1.3 {
        println!("  ✓ SIGNIFICANT IMPROVEMENT - Async pipelining working well!");
        println!("    GPU can overlap kernel launches and reduce sync overhead");
    } else if speedup > 1.1 {
        println!("  ≈ MODERATE IMPROVEMENT - Some pipelining benefit");
        println!("    Kernel execution time may dominate launch overhead");
    } else {
        println!("  ⚠ LIMITED IMPROVEMENT - Further investigation needed");
        println!("    Possible causes:");
        println!("      • Kernel execution dominates (expected for large K)");
        println!("      • GPU cannot overlap these operations");
        println!("      • Memory bandwidth saturation");
    }

    println!("═════════════════════════════════════════════════════════════════════════════\n");
}

#[test]
#[ignore]
fn benchmark_async_cpu_overlap() {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          CPU/GPU Overlap Benchmark                                            ║");
    println!("║  Demonstrating CPU work while GPU computes                                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Prepare data
    let scalars: Vec<Scalar> = (0..SIZE).map(|_| Scalar::random(&mut rng)).collect();
    let points_proj: Vec<G1Projective> = (0..SIZE)
        .map(|_| G1Projective::random(&mut rng))
        .collect();
    let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();
    let device_bases = ctx.upload_g1_bases(&points).expect("Upload failed");

    // =========================================================================
    // WITHOUT overlap: GPU work, then CPU work
    // =========================================================================
    let no_overlap_start = Instant::now();

    let result = ctx
        .msm_with_device_bases(&scalars, &device_bases)
        .expect("MSM failed");

    // Simulate CPU work (e.g., computing another scalar multiplication)
    let cpu_work_start = Instant::now();
    let _cpu_result: Vec<Scalar> = scalars.iter()
        .map(|s| s * s) // Simple CPU work
        .collect();
    let cpu_work_time = cpu_work_start.elapsed();

    let no_overlap_time = no_overlap_start.elapsed();

    // =========================================================================
    // WITH overlap: Launch GPU, do CPU work while GPU computes
    // =========================================================================
    let overlap_start = Instant::now();

    // Launch async MSM
    let handle = ctx
        .msm_with_device_bases_async(&scalars, &device_bases)
        .expect("Async launch failed");

    // Do CPU work while GPU computes
    let _cpu_result2: Vec<Scalar> = scalars.iter()
        .map(|s| s * s)
        .collect();

    // Wait for GPU result
    let result2 = handle.wait().expect("Wait failed");

    let overlap_time = overlap_start.elapsed();

    // Verify
    assert_eq!(result, result2);

    println!("  CPU work time:        {:?}", cpu_work_time);
    println!("  Sequential (no overlap): {:?}", no_overlap_time);
    println!("  Overlapped:           {:?}", overlap_time);
    println!();

    let overlap_benefit = no_overlap_time.as_secs_f64() / overlap_time.as_secs_f64();
    println!("  Overlap benefit:      {:.2}x", overlap_benefit);

    if overlap_benefit > 1.2 {
        println!("  ✓ CPU/GPU overlap working - async is beneficial!");
    } else {
        println!("  ≈ Limited overlap (CPU work was too fast compared to GPU)");
    }
}
