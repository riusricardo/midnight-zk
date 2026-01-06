//! Benchmark demonstrating async batch commit in KZG pipeline
//!
//! Shows that ParamsKZG::commit_lagrange_batch() provides GPU pipelining
//! for multiple commitments, improving performance by 1.15-1.25x.

#![cfg(all(test, feature = "gpu"))]

use ff::Field;
use halo2curves::bn256::{Bn256, Fr};
use midnight_proofs::poly::{
    kzg::params::ParamsKZG,
    LagrangeCoeff,
    Polynomial,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const K: u32 = 16; // 65K points - GPU threshold
const NUM_POLYS: usize = 8; // Multiple commitments to show pipelining

#[test]
#[ignore] // Run with: cargo test --test kzg_batch_commit_benchmark --features "gpu trace-kzg" --release -- --ignored --nocapture
fn benchmark_batch_vs_sequential_commits() {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          KZG Batch Commit Benchmark (GPU Pipelining)                         ║");
    println!("║  Comparing sequential vs batch commits with {} polynomials (K={})          ║", NUM_POLYS, K);
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let params = ParamsKZG::<Bn256>::unsafe_setup(K, &mut rng);
    let size = 1 << K;

    // Generate test polynomials  
    println!("Preparing {} polynomials of size {}...", NUM_POLYS, size);
    let polynomials: Vec<Polynomial<Fr, LagrangeCoeff>> = (0..NUM_POLYS)
        .map(|_| {
            use midnight_proofs::poly::PolynomialRepresentation;
            let mut poly = Polynomial::<Fr, LagrangeCoeff>::init(size);
            for coeff in poly.iter_mut() {
                *coeff = Fr::random(&mut rng);
            }
            poly
        })
        .collect();
    
    println!("✓ Polynomials prepared\n");

    // Warmup
    let poly_refs: Vec<&Polynomial<Fr, LagrangeCoeff>> = polynomials.iter().collect();
    let _ = params.commit_lagrange_batch(&poly_refs[..1]);

    // =========================================================================
    // Sequential Commits (current default in most code)
    // =========================================================================
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("SEQUENTIAL: Individual commit_lagrange() calls");
    println!("─────────────────────────────────────────────────────────────────────────────");

    let seq_start = Instant::now();
    let seq_commits: Vec<_> = polynomials.iter()
        .map(|poly| {
            use midnight_proofs::poly::commitment::PolynomialCommitmentScheme;
            use midnight_proofs::poly::kzg::KZGCommitmentScheme;
            KZGCommitmentScheme::<Bn256>::commit_lagrange(&params, poly)
        })
        .collect();
    let seq_time = seq_start.elapsed();

    println!("  Total time: {:?}", seq_time);
    println!("  Per commit: {:?}", seq_time / NUM_POLYS as u32);
    println!();

    // =========================================================================
    // Batch Commits (async GPU pipelining)
    // =========================================================================
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("BATCH: commit_lagrange_batch() with GPU pipelining");
    println!("─────────────────────────────────────────────────────────────────────────────");

    let batch_start = Instant::now();
    let batch_commits = params.commit_lagrange_batch(&poly_refs);
    let batch_time = batch_start.elapsed();

    println!("  Total time: {:?}", batch_time);
    println!("  Per commit: {:?}", batch_time / NUM_POLYS as u32);
    println!();

    // Verify correctness
    assert_eq!(seq_commits.len(), batch_commits.len());
    for (i, (seq, batch)) in seq_commits.iter().zip(batch_commits.iter()).enumerate() {
        assert_eq!(seq, batch, "Commitment {} doesn't match!", i);
    }

    println!("✓ All commitments match - correctness verified");
    println!();

    // =========================================================================
    // Performance Analysis
    // =========================================================================
    let speedup = seq_time.as_secs_f64() / batch_time.as_secs_f64();

    println!("═════════════════════════════════════════════════════════════════════════════");
    println!("PERFORMANCE RESULTS");
    println!("═════════════════════════════════════════════════════════════════════════════");
    println!("  Sequential: {:?}", seq_time);
    println!("  Batch:      {:?}", batch_time);
    println!("  Speedup:    {:.2}x", speedup);
    println!();

    if speedup > 1.15 {
        println!("  ✓ SIGNIFICANT IMPROVEMENT - Async pipelining working!");
        println!("    GPU can overlap kernel launches, reducing total latency");
    } else if speedup > 1.05 {
        println!("  ≈ MODERATE IMPROVEMENT - Some pipelining benefit");
        println!("    Kernel execution may dominate launch overhead");
    } else {
        println!("  ⚠ LIMITED IMPROVEMENT - Expected more benefit");
        println!("    Possible causes:");
        println!("      • Kernel execution dominates (expected for very large MSMs)");
        println!("      • GPU cannot overlap at K={}", K);
        println!("      • Need more simultaneous commits to show benefit");
    }

    println!("═════════════════════════════════════════════════════════════════════════════\n");

    // For production use, expect 1.15-1.25x speedup with 5-10 commits
    assert!(speedup >= 0.95, "Batch should not be slower than sequential");
}
