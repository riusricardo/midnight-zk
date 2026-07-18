//! GPU MSM Performance Benchmark
//!
//! This benchmark tests dense polynomial commitments where GPU acceleration
//! provides actual speedups. Each test runs twice - first run includes SRS
//! base upload overhead, second run measures actual cached performance.
//!
//! Run with:
//!   cargo test --test gpu_msm_benchmark --features gpu --release -- --ignored --nocapture

#![cfg(feature = "gpu")]

use ff::Field;
use group::{Curve, Group};
use midnight_curves::{Fq as Scalar, G1Affine, G1Projective};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use midnight_bls12_381_cuda::{GpuMsmContext, should_use_gpu, device_type};

/// Benchmark MSM at a given size, comparing GPU vs CPU
/// 
/// Uses the optimized path: upload bases in Montgomery form, use msm_with_device_bases()
/// This is the same path KZG uses for polynomial commitments.
/// 
/// Runs GPU MSM twice:
/// - First run: includes base upload to GPU memory (warmup)
/// - Second run: bases already cached, measures actual MSM performance
/// 
/// Returns: (gpu_warmup_ms, gpu_cached_ms, cpu_ms, speedup)
fn benchmark_msm(size: usize, ctx: &GpuMsmContext) -> (f64, f64, f64, f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(size as u64);
    
    // Generate random scalars and points
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let points_proj: Vec<G1Projective> = (0..size)
        .map(|_| G1Projective::random(&mut rng))
        .collect();
    let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();
    
    // Optimized path: upload bases in Montgomery form (zero-copy conversion)
    let warmup_start = Instant::now();
    let device_bases = ctx.upload_g1_bases(&points).expect("Failed to upload bases");
    
    // Run 1: First MSM with freshly uploaded bases
    let _ = ctx.msm_with_device_bases(&scalars, &device_bases).expect("GPU MSM failed");
    let gpu_warmup = warmup_start.elapsed().as_secs_f64() * 1000.0;
    
    // Run 2: GPU with cached bases (accurate performance measurement)
    let gpu_start = Instant::now();
    let gpu_result = ctx.msm_with_device_bases(&scalars, &device_bases).expect("GPU MSM failed");
    let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;
    
    // Benchmark CPU MSM (BLST)
    let cpu_start = Instant::now();
    let cpu_result = G1Projective::multi_exp(&points_proj, &scalars);
    let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;
    
    // Verify results match
    assert_eq!(gpu_result, cpu_result, "GPU and CPU results must match!");
    
    let speedup = cpu_time / gpu_time;
    (gpu_warmup, gpu_time, cpu_time, speedup)
}

#[test]
#[ignore]
fn gpu_msm_benchmark() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  GPU MSM Performance Benchmark - Dense Polynomial Commitments                 ║");
    println!("║  Each size runs twice: warmup (includes upload) + cached (accurate timing)    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Check GPU availability
    println!("Configuration:");
    println!("  • Device type: {}", device_type());
    println!("  • GPU threshold: {} points (K ≥ {})", 
             midnight_bls12_381_cuda::min_gpu_size(),
             midnight_bls12_381_cuda::min_gpu_size().trailing_zeros());
    
    // Initialize GPU context
    print!("  • Initializing GPU... ");
    let init_start = Instant::now();
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    println!("done in {:?}", init_start.elapsed());
    
    // CUDA context warmup
    print!("  • CUDA context warmup... ");
    let warmup_start = Instant::now();
    let _ = ctx.warmup();
    println!("done in {:?}", warmup_start.elapsed());
    
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    
    // Test sizes: K=14-20 representing realistic Midnight circuit sizes
    let test_sizes = vec![
        (14, "16K"),
        (15, "32K"),
        (16, "64K  ← threshold"),
        (17, "128K"),
        (18, "256K"),
        (19, "512K"),
        (20, "1M"),
    ];
    
    println!("┌───────┬──────────┬────────────┬────────────┬────────────┬──────────┬─────────────┐");
    println!("│   K   │  Points  │ Warm (ms)  │ Cache (ms) │  CPU (ms)  │ Speedup  │   Status    │");
    println!("├───────┼──────────┼────────────┼────────────┼────────────┼──────────┼─────────────┤");
    
    let mut results = Vec::new();
    
    for (k, label) in test_sizes {
        let size = 1usize << k;
        
        let (gpu_warmup, gpu_cached, cpu_time, speedup) = benchmark_msm(size, &ctx);
        
        let status = if speedup > 1.5 {
            "✓ GPU wins"
        } else if speedup > 0.9 {
            "≈ Similar"
        } else {
            "✗ CPU wins"
        };
        
        let gpu_better = should_use_gpu(size);
        let threshold_marker = if gpu_better { "→" } else { " " };
        
        println!("│{} K={:<2} │ {:>8} │ {:>10.2} │ {:>10.2} │ {:>10.2} │ {:>7.2}x │ {:>11} │",
                 threshold_marker, k, label, gpu_warmup, gpu_cached, cpu_time, speedup, status);
        
        results.push((k, size, gpu_cached, cpu_time, speedup));
    }
    
    println!("└───────┴──────────┴────────────┴────────────┴────────────┴──────────┴─────────────┘");
    println!();
    println!("Legend: → = Above GPU threshold (will use GPU in Auto mode)");
    println!("        Warm = First run (includes SRS upload), Cache = Second run (bases cached)");
    println!();
    
    // Summary
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("📊 Analysis:");
    
    // Find crossover point
    let crossover = results.iter()
        .find(|(_, _, gpu, cpu, _)| gpu < cpu)
        .map(|(k, _, _, _, _)| *k);
    
    if let Some(k) = crossover {
        println!("  • GPU becomes faster at K={} ({} points)", k, 1 << k);
    }
    
    // Best speedup
    if let Some((k, _size, _, _, speedup)) = results.iter()
        .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
    {
        println!("  • Best GPU speedup: {:.2}x at K={} ({} points)", speedup, k, 1 << k);
    }
    
    // Largest tested
    if let Some((k, _size, gpu, cpu, speedup)) = results.last() {
        println!("  • Largest size (K={}): GPU={:.1}ms, CPU={:.1}ms, speedup={:.2}x",
                 k, gpu, cpu, speedup);
    }
    
    println!();
    println!("💡 Recommendation:");
    println!("  • Use GPU for MSMs with ≥64K points (K≥16)");
    println!("  • Use BLST (CPU) for smaller MSMs due to GPU transfer overhead");
    println!();
}

/// Quick benchmark for CI - tests K=14,15,16,17,18,19 with accurate warmup/cached timing
#[test]
#[ignore]
fn gpu_msm_benchmark_quick() {
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let _ = ctx.warmup();
    
    println!("\nQuick GPU MSM Benchmark (2nd run = cached, bases in GPU memory):");
    println!("────────────────────────────────────────────────────────────────");
    println!("  K  │ Warm (ms) │ Cache (ms) │ CPU (ms) │ Speedup");
    println!("─────┼───────────┼────────────┼──────────┼─────────");
    
    for k in [14, 15, 16, 17, 18, 19] {
        let size = 1usize << k;
        let (warmup, cached, cpu, speedup) = benchmark_msm(size, &ctx);
        let marker = if speedup > 1.0 { "✓" } else { " " };
        println!(" {:>3} │ {:>9.1} │ {:>10.1} │ {:>8.1} │ {:>5.2}x {}",
                 k, warmup, cached, cpu, speedup, marker);
    }
    
    println!("────────────────────────────────────────────────────────────────");
    println!("Legend: Warm = includes base upload, Cache = bases already in GPU memory");
}

/// Test that GPU results are correct (using optimized path)
#[test]
fn gpu_msm_correctness() {
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");

    for k in [10, 12, 14, 16] {
        let size = 1usize << k;
        let mut rng = ChaCha8Rng::seed_from_u64(k as u64);

        let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
        let points_proj: Vec<G1Projective> = (0..size)
            .map(|_| G1Projective::random(&mut rng))
            .collect();
        let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();

        // Test optimized path: upload bases then run MSM
        let device_bases = ctx.upload_g1_bases(&points).expect("Failed to upload bases");
        let gpu_result = ctx.msm_with_device_bases(&scalars, &device_bases).expect("GPU MSM failed");
        let cpu_result = G1Projective::multi_exp(&points_proj, &scalars);

        assert_eq!(gpu_result, cpu_result, "GPU/CPU mismatch at K={}", k);
    }

    println!("✓ GPU MSM correctness verified for K=10,12,14,16");
}

/// Degenerate scalar distributions that stress bucket accumulation:
/// - all-equal scalars: every window's contributions land in ONE bucket
///   (maximal mega-buckets / partial chains)
/// - all-zero scalars: every digit is a zero digit (all-INVALID stream)
/// - zero-padded tail: realistic prover shape (dense prefix, zero tail)
/// - tiny scalars: only the lowest window is populated
#[test]
fn gpu_msm_correctness_degenerate_scalars() {
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let k = 16;
    let size = 1usize << k;
    let mut rng = ChaCha8Rng::seed_from_u64(0xD15C0);

    let points_proj: Vec<G1Projective> = (0..size)
        .map(|_| G1Projective::random(&mut rng))
        .collect();
    let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();
    let device_bases = ctx.upload_g1_bases(&points).expect("Failed to upload bases");

    let one = Scalar::ONE;
    let some = Scalar::random(&mut rng);
    let cases: Vec<(&str, Vec<Scalar>)> = vec![
        ("all-equal", vec![some; size]),
        ("all-zero", vec![Scalar::ZERO; size]),
        ("all-one", vec![one; size]),
        ("zero-padded-tail", {
            let mut v: Vec<Scalar> = (0..size / 4).map(|_| Scalar::random(&mut rng)).collect();
            v.resize(size, Scalar::ZERO);
            v
        }),
        ("tiny", (0..size).map(|i| Scalar::from(i as u64 % 7)).collect()),
    ];

    for (name, scalars) in cases {
        let gpu_result = ctx
            .msm_with_device_bases(&scalars, &device_bases)
            .expect("GPU MSM failed");
        let cpu_result = G1Projective::multi_exp(&points_proj, &scalars);
        assert_eq!(gpu_result, cpu_result, "GPU/CPU mismatch at K={} case={}", k, name);
        println!("  ✓ {}", name);
    }

    println!("✓ GPU MSM degenerate-scalar correctness verified at K=16");
}

/// Duplicate/repeated base points: the only realistic trigger of the P+P
/// (doubling) and P+(-P) (infinity) edge cases inside bucket accumulation —
/// signed digits can put +P and -P for the same base into one bucket.
#[test]
fn gpu_msm_correctness_duplicate_points() {
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let k = 14;
    let size = 1usize << k;
    let mut rng = ChaCha8Rng::seed_from_u64(0xD0B1E);

    // A quarter of the points are copies of point 0; another quarter copies
    // of point 1 — guarantees many same-base collisions inside buckets.
    let mut points_proj: Vec<G1Projective> = (0..size)
        .map(|_| G1Projective::random(&mut rng))
        .collect();
    for i in 0..size / 4 {
        points_proj[i] = points_proj[0];
    }
    for i in size / 4..size / 2 {
        points_proj[i] = points_proj[size / 2];
    }
    let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();
    let device_bases = ctx.upload_g1_bases(&points).expect("Failed to upload bases");

    // Case 1: random scalars over duplicated points
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let gpu = ctx.msm_with_device_bases(&scalars, &device_bases).expect("GPU MSM failed");
    let cpu = G1Projective::multi_exp(&points_proj, &scalars);
    assert_eq!(gpu, cpu, "duplicate-points random-scalar mismatch");

    // Case 2: engineered +P/-P collisions — scalar s and r-s put +P and -P
    // digit contributions of the SAME base into mirrored buckets.
    let s = Scalar::random(&mut rng);
    let neg_s = -s;
    let scalars2: Vec<Scalar> = (0..size)
        .map(|i| if i % 2 == 0 { s } else { neg_s })
        .collect();
    let gpu2 = ctx.msm_with_device_bases(&scalars2, &device_bases).expect("GPU MSM failed");
    let cpu2 = G1Projective::multi_exp(&points_proj, &scalars2);
    assert_eq!(gpu2, cpu2, "duplicate-points +/-s mismatch");

    println!("✓ GPU MSM duplicate-point edge cases verified at K=14");
}

/// Host-side simulation of the chunked-accumulation partial-chain distribution.
/// Replicates the CUDA signed-digit decomposition (c=15, 18 windows, zero->skip)
/// for the exact PROFILE_K scalars and reports how many boundary-partial chains
/// the combine pass must walk, their total adds, and the longest chain.
#[test]
#[ignore]
fn sim_chain_distribution() {
    use ff::PrimeField;
    let k: usize = std::env::var("PROFILE_K").unwrap_or_else(|_| "19".into()).parse().unwrap();
    let size = 1usize << k;
    let c: i64 = 15;
    let windows: usize = 18; // ceil(256/15) with bitsize = LIMBS*64
    let nb: i64 = 1 << (c - 1);
    let bpw = (nb + 1) as usize;

    let mut rng = ChaCha8Rng::seed_from_u64(k as u64);
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();

    let mut counts = vec![0u32; windows * bpw];
    let mut zeros = 0u64;
    for s in &scalars {
        let repr = s.to_repr();
        let bytes: &[u8] = repr.as_ref();
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        let mut carry: i64 = 0;
        for w in 0..windows {
            let bit = (w as i64) * c;
            let li = (bit / 64) as usize;
            let off = (bit % 64) as u32;
            let mut window = limbs[li] >> off;
            if off as i64 + c > 64 && li + 1 < 4 {
                window |= limbs[li + 1] << (64 - off);
            }
            window &= (1u64 << c) - 1;
            let mut val = window as i64 + carry;
            carry = 0;
            if val > nb {
                val = (1i64 << c) - val;
                carry = 1;
            }
            if val == 0 {
                zeros += 1;
            } else {
                counts[w * bpw + (val as usize - 1)] += 1;
            }
        }
        assert_eq!(carry, 0, "final carry must not occur for valid scalars");
    }

    let chunk: u64 = 128;
    let mut pos: u64 = 0;
    let (mut nchains, mut total_adds, mut max_chain) = (0u64, 0u64, 0u64);
    let mut top_buckets: Vec<u32> = counts.iter().copied().collect();
    top_buckets.sort_unstable_by(|a, b| b.cmp(a));
    for &cnt in &counts {
        if cnt == 0 { continue; }
        let start = pos;
        let end = pos + cnt as u64;
        pos = end;
        let spans = (end - 1) / chunk - start / chunk + 1;
        if spans >= 2 {
            nchains += 1;
            let adds = spans - 1;
            total_adds += adds;
            max_chain = max_chain.max(adds);
        }
    }
    println!(
        "K={} valid={} zeros={} | chains={} total_chain_adds={} max_chain={} | top buckets: {:?}",
        k, pos, zeros, nchains, total_adds, max_chain, &top_buckets[..8]
    );
}

/// Profiling test - runs a single MSM at K specified by PROFILE_K env var
/// Uses the optimized path (upload_g1_bases + msm_with_device_bases)
/// Use with: PROFILE_K=16 ncu --set full ./target/release/deps/gpu_msm_benchmark-* profile_msm_single
#[test]
#[ignore]
fn profile_msm_single() {
    let k: usize = std::env::var("PROFILE_K")
        .unwrap_or_else(|_| "16".to_string())
        .parse()
        .expect("PROFILE_K must be a number");
    
    let size = 1usize << k;
    println!("Profiling MSM at K={} ({} points) using optimized path", k, size);
    
    let mut rng = ChaCha8Rng::seed_from_u64(k as u64);
    
    let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
    let points_proj: Vec<G1Projective> = (0..size)
        .map(|_| G1Projective::random(&mut rng))
        .collect();
    let points: Vec<G1Affine> = points_proj.iter().map(|p| p.to_affine()).collect();
    
    let ctx = GpuMsmContext::new().expect("Failed to create GPU context");
    let _ = ctx.warmup();
    
    // Upload bases in Montgomery form (zero-copy)
    println!("Uploading bases to GPU (Montgomery form)...");
    let upload_start = Instant::now();
    let device_bases = ctx.upload_g1_bases(&points).expect("Failed to upload bases");
    println!("Bases uploaded in {:?}", upload_start.elapsed());
    
    // Warmup run (cached bases)
    println!("Warmup run...");
    let _ = ctx.msm_with_device_bases(&scalars, &device_bases).expect("GPU MSM failed");
    
    // Profiled run (cached bases, warmed up)
    println!("Profiled run (bases cached, kernel warmed up)...");
    let start = Instant::now();
    let result = ctx.msm_with_device_bases(&scalars, &device_bases).expect("GPU MSM failed");
    let elapsed = start.elapsed();
    
    println!("MSM completed in {:?}", elapsed);
    println!("Result (first bytes): {:?}", &format!("{:?}", result)[..80]);
}
