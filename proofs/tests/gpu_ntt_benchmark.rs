//! GPU NTT/FFT benchmark + correctness gate.
//!
//! Mirrors the prover's actual dispatch path (host slices, in-place, via
//! `forward_ntt_inplace_auto` / `inverse_ntt_inplace_auto` — the same calls
//! `utils::fft::{best_fft, ifft}` route to), so timings include H2D/D2H.
//! Every timed run is also compared against the CPU reference — a benchmark
//! row is simultaneously a correctness assertion.
//!
//! Run with:
//!   cargo test --test gpu_ntt_benchmark --features gpu --release -- --ignored --nocapture

#![cfg(feature = "gpu")]

use ff::{Field, PrimeField};
use midnight_curves::Fq as Scalar;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn omega_for(k: u32) -> Scalar {
    let mut omega = Scalar::ROOT_OF_UNITY;
    for _ in k..Scalar::S {
        omega = omega.square();
    }
    omega
}

fn cpu_fft(data: &mut [Scalar], omega: Scalar, k: u32) {
    midnight_curves::fft::best_fft(data, omega, k);
}

/// (gpu_fwd_ms, gpu_inv_ms, cpu_fwd_ms) medians of `reps` runs; asserts
/// GPU forward == CPU forward and GPU roundtrip == identity every rep.
fn bench_size(k: u32, reps: usize) -> (f64, f64, f64) {
    let n = 1usize << k;
    let omega = omega_for(k);
    let omega_inv = omega.invert().unwrap();
    let divisor = Scalar::from(n as u64).invert().unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(k as u64);
    let original: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();

    let mut gpu_fwd = Vec::with_capacity(reps);
    let mut gpu_inv = Vec::with_capacity(reps);
    let mut cpu_fwd = Vec::with_capacity(reps);

    for _ in 0..reps {
        // GPU forward
        let mut g = original.clone();
        let t = Instant::now();
        midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut g).expect("GPU NTT failed");
        gpu_fwd.push(t.elapsed().as_secs_f64() * 1000.0);

        // CPU forward (reference + timing)
        let mut c = original.clone();
        let t = Instant::now();
        cpu_fft(&mut c, omega, k);
        cpu_fwd.push(t.elapsed().as_secs_f64() * 1000.0);

        assert_eq!(g, c, "GPU forward NTT != CPU at K={}", k);

        // GPU inverse (includes 1/n): roundtrip must give back the input
        let t = Instant::now();
        midnight_bls12_381_cuda::ntt::inverse_ntt_inplace_auto(&mut g).expect("GPU iNTT failed");
        gpu_inv.push(t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(g, original, "GPU NTT roundtrip != identity at K={}", k);

        // CPU inverse reference: must also match the GPU inverse of the
        // forward result (validates 1/n handling, not just roundtrip)
        cpu_fft(&mut c, omega_inv, k);
        for v in c.iter_mut() {
            *v *= divisor;
        }
        assert_eq!(c, original, "CPU inverse reference broken at K={}", k);
    }

    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    (med(&mut gpu_fwd), med(&mut gpu_inv), med(&mut cpu_fwd))
}

/// Impulse-response check: input = e_j must transform to out[i] = omega^(i*j).
/// This pins down ordering AND twiddle correctness exactly (a permuted output
/// or wrong twiddle basis cannot pass for multiple j). K values chosen to
/// exercise every tiled high-pass width: K=12 (r=2), 13 (r=3), 16 (r=5+1),
/// 17 (r=5+2), plus the pure low-pass edge at K=12..13.
#[test]
fn gpu_ntt_impulse_response() {
    for k in [12u32, 13, 16, 17] {
        let n = 1usize << k;
        let omega = omega_for(k);
        for j in [0usize, 1, 7, n - 1] {
            let mut data = vec![Scalar::ZERO; n];
            data[j] = Scalar::ONE;
            midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut data)
                .expect("GPU NTT failed");

            let wj = omega.pow_vartime([j as u64]);
            let mut expect = Scalar::ONE;
            for (i, v) in data.iter().enumerate() {
                assert_eq!(*v, expect, "impulse mismatch at k={} j={} i={}", k, j, i);
                expect *= wj;
            }

            // and back: inverse must recover e_j exactly
            midnight_bls12_381_cuda::ntt::inverse_ntt_inplace_auto(&mut data)
                .expect("GPU iNTT failed");
            for (i, v) in data.iter().enumerate() {
                let want = if i == j { Scalar::ONE } else { Scalar::ZERO };
                assert_eq!(*v, want, "inverse impulse mismatch at k={} j={} i={}", k, j, i);
            }
        }
    }
    println!("✓ NTT impulse response verified (K=12,13,16,17 × j=0,1,7,n-1)");
}

/// Batched NTT must produce exactly the same transforms as one-by-one calls.
#[test]
fn gpu_ntt_batch_matches_singles() {
    let k = 14u32;
    let n = 1usize << k;
    let b = 8usize;
    let mut rng = ChaCha8Rng::seed_from_u64(0xBA7C4);
    let data: Vec<Scalar> = (0..n * b).map(|_| Scalar::random(&mut rng)).collect();

    let batched = midnight_bls12_381_cuda::ntt::forward_ntt_batch_auto(&data, n)
        .expect("batched NTT failed");
    assert_eq!(batched.len(), n * b);

    for i in 0..b {
        let mut single = data[i * n..(i + 1) * n].to_vec();
        midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut single)
            .expect("single NTT failed");
        assert_eq!(&batched[i * n..(i + 1) * n], &single[..], "batch element {} differs", i);
    }

    let inv = midnight_bls12_381_cuda::ntt::inverse_ntt_batch_auto(&batched, n)
        .expect("batched iNTT failed");
    assert_eq!(inv, data, "batched roundtrip != identity");

    println!("✓ batched NTT == singles and roundtrips (b={} × 2^{})", b, k);
}

/// Batched call vs host loop of singles (same total work) — measures what
/// grid-level batching buys the prover's many same-size column transforms.
#[test]
#[ignore]
fn gpu_ntt_batch_benchmark() {
    let k = 18u32;
    let n = 1usize << k;
    let b = 16usize;
    let mut rng = ChaCha8Rng::seed_from_u64(k as u64);
    let data: Vec<Scalar> = (0..n * b).map(|_| Scalar::random(&mut rng)).collect();

    // Full-size warmup for BOTH arms: domain init, mem-pool growth to the
    // batch working set, and host page-faults must land outside the timings.
    let _ = midnight_bls12_381_cuda::ntt::forward_ntt_batch_auto(&data, n).unwrap();
    let mut warm = data.clone();
    for i in 0..b {
        midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut warm[i * n..(i + 1) * n])
            .unwrap();
    }

    let t = Instant::now();
    let batched = midnight_bls12_381_cuda::ntt::forward_ntt_batch_auto(&data, n).unwrap();
    let batched_ms = t.elapsed().as_secs_f64() * 1000.0;

    let mut loop_out = data.clone();
    let t = Instant::now();
    for i in 0..b {
        midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut loop_out[i * n..(i + 1) * n])
            .unwrap();
    }
    let loop_ms = t.elapsed().as_secs_f64() * 1000.0;

    // In-place batched: no output Vec allocation / zeroing / copy-back —
    // the API the prover integration should use.
    let ctx = midnight_bls12_381_cuda::ntt::GpuNttContext::new(24).unwrap();
    let mut inplace_out = data.clone();
    ctx.forward_ntt_batch_inplace(&mut inplace_out, n).unwrap(); // warm
    inplace_out.copy_from_slice(&data);
    let t = Instant::now();
    ctx.forward_ntt_batch_inplace(&mut inplace_out, n).unwrap();
    let inplace_ms = t.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(batched, loop_out, "batch and loop disagree");
    assert_eq!(inplace_out, loop_out, "in-place batch and loop disagree");
    println!(
        "batch {}×2^{}: batched(Vec) {:.2} ms | batched(in-place) {:.2} ms | loop-of-singles {:.2} ms",
        b, k, batched_ms, inplace_ms, loop_ms
    );
}

#[test]
#[ignore]
fn gpu_ntt_benchmark() {
    // Domain init + CUDA context warmup (paid once, outside timings)
    {
        let mut warm: Vec<Scalar> = (0..4096).map(|i| Scalar::from(i as u64)).collect();
        midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut warm).expect("warmup failed");
    }

    println!("\nGPU NTT Benchmark (host->host in-place, prover dispatch path; median of 3):");
    println!("  K  │ GPU fwd (ms) │ GPU inv (ms) │ CPU fwd (ms) │ Speedup(fwd)");
    println!("─────┼──────────────┼──────────────┼──────────────┼─────────────");

    let ks: Vec<u32> = std::env::var("NTT_BENCH_KS")
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![12, 14, 16, 18, 20, 21, 22]);
    for k in ks {
        let (gf, gi, cf) = bench_size(k, 3);
        let marker = if cf / gf > 1.0 { "✓" } else { " " };
        println!(
            " {:>3} │ {:>12.2} │ {:>12.2} │ {:>12.2} │ {:>6.2}x {}",
            k, gf, gi, cf, cf / gf, marker
        );
    }
    println!("\nAll rows correctness-checked: GPU fwd == CPU fwd, GPU roundtrip == identity.");
}

/// FAIR NTT comparison (ours vs closed): timing + roundtrip only, NO CPU-basis
/// assert. Run with MIDNIGHT_NTT_ICICLE_ROU=1 so the domain uses ICICLE's own
/// root — the only root the CLOSED backend accepts — letting both backends be
/// timed on identical NTT work. The roundtrip inverse(forward(x))==x is
/// root-agnostic, so it still verifies correctness for whichever backend runs.
///
///   MIDNIGHT_NTT_ICICLE_ROU=1 NTT_BENCH_KS=16,18,20,22 \
///     ICICLE_BACKEND_INSTALL_DIR=<backend> \
///     cargo test --test gpu_ntt_benchmark --features gpu --release -- \
///     gpu_ntt_fair_bench --ignored --nocapture
#[test]
#[ignore = "fair ours-vs-closed NTT bench; needs MIDNIGHT_NTT_ICICLE_ROU=1"]
fn gpu_ntt_fair_bench() {
    if !midnight_bls12_381_cuda::is_gpu_available() {
        eprintln!("[skip] no GPU");
        return;
    }
    let icicle_rou = std::env::var("MIDNIGHT_NTT_ICICLE_ROU").is_ok();
    let ks: Vec<u32> = std::env::var("NTT_BENCH_KS")
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![16, 18, 20, 22]);
    println!(
        "\nFAIR NTT bench  backend={}  root={}",
        std::env::var("ICICLE_BACKEND_INSTALL_DIR").unwrap_or_else(|_| "<default>".into()),
        if icicle_rou { "ICICLE" } else { "midnight" },
    );
    println!("{:>4} {:>12} {:>12} {:>10}", "K", "fwd(ms)", "inv(ms)", "roundtrip");
    let reps = 5usize;
    for k in ks {
        let n = 1usize << k;
        let mut rng = ChaCha8Rng::seed_from_u64(k as u64);
        let orig: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();

        // warmup (also builds/caches the domain + bases)
        let mut w = orig.clone();
        midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut w).unwrap();
        midnight_bls12_381_cuda::ntt::inverse_ntt_inplace_auto(&mut w).unwrap();

        let mut fwd_best = f64::MAX;
        let mut inv_best = f64::MAX;
        let mut roundtrip_ok = true;
        for _ in 0..reps {
            let mut data = orig.clone();
            let t = Instant::now();
            midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(&mut data).unwrap();
            fwd_best = fwd_best.min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            midnight_bls12_381_cuda::ntt::inverse_ntt_inplace_auto(&mut data).unwrap();
            inv_best = inv_best.min(t.elapsed().as_secs_f64() * 1e3);
            if data != orig {
                roundtrip_ok = false;
            }
        }
        println!(
            "{k:>4} {fwd_best:>12.3} {inv_best:>12.3} {:>10}",
            if roundtrip_ok { "OK" } else { "FAIL" }
        );
        assert!(roundtrip_ok, "NTT roundtrip identity failed at k={k}");
    }
}
