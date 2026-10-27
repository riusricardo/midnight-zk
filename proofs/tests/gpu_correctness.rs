//! GPU correctness tests (GATE 0).
//!
//! These assert that enabling GPU acceleration never changes a computed result: GPU NTTs,
//! KZG commitments, and full proofs must equal their CPU (midnight_curves / BLST)
//! counterparts. They exist because the restored fork's GPU proving was UNSOUND and no
//! test caught it:
//!   * C1 — `EvaluationDomain::ifft` ran a FORWARD NTT on the GPU (discarded `omega_inv`).
//!   * R1 — the GPU NTT domain was seeded with ICICLE's own root of unity, a different
//!     primitive 2^32-th root than `midnight_curves::Fq::ROOT_OF_UNITY`, so every GPU NTT
//!     was a permutation of the CPU basis.
//! The existing `e2e_proof_benchmark` created proofs but never VERIFIED them, so both bugs
//! were invisible. `gpu_full_proof_verifies` below closes that gap.
//!
//! Run against ours vs the closed backend via the `ICICLE_BACKEND_INSTALL_DIR` A/B switch:
//!   ICICLE_BACKEND_INSTALL_DIR=~/.local/icicle-backends/ours \
//!     cargo test --test gpu_correctness --features gpu --release -- --nocapture

#![cfg(feature = "gpu")]

use ff::{Field, PrimeField};
use midnight_curves::{Bls12, Fq, G1Projective};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use midnight_proofs::poly::{
    commitment::PolynomialCommitmentScheme,
    kzg::{params::ParamsKZG, KZGCommitmentScheme},
    EvaluationDomain,
};

/// Skip (rather than fail) when no usable GPU is present, so the suite is safe to run on
/// CI hosts without a CUDA device.
fn gpu_present() -> bool {
    if midnight_bls12_381_cuda::is_gpu_available() {
        true
    } else {
        eprintln!("[skip] GPU backend not available; skipping GPU correctness test");
        false
    }
}

/// The 2^k-th root of unity in midnight_curves' convention (matches EvaluationDomain).
fn omega_for(k: u32) -> Fq {
    let mut omega = Fq::ROOT_OF_UNITY;
    for _ in k..Fq::S {
        omega = omega.square();
    }
    omega
}

/// R1 canary: GPU forward NTT (`coeff_to_lagrange`) must equal CPU `best_fft` element-wise.
/// This fails if the GPU domain is seeded with a root that differs from midnight_curves'.
#[test]
fn gpu_forward_ntt_matches_cpu() {
    if !gpu_present() {
        return;
    }
    for k in [12u32, 14, 16] {
        let n = 1usize << k;
        let domain = EvaluationDomain::<Fq>::new(1, k);
        let mut rng = ChaCha8Rng::seed_from_u64(k as u64);
        let coeffs: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();

        // GPU path: coeff_to_lagrange runs a forward NTT via the GPU dispatcher at k >= 12.
        let gpu = domain.coeff_to_lagrange(domain.coeff_from_vec(coeffs.clone()));

        // CPU reference: raw midnight_curves best_fft with the domain's own omega.
        let mut cpu = coeffs.clone();
        midnight_curves::fft::best_fft(&mut cpu, domain.get_omega(), k);

        assert_eq!(&*gpu, cpu.as_slice(), "GPU forward NTT != CPU best_fft at k={k}");
    }
}

/// C1 canary: GPU inverse NTT (`lagrange_to_coeff`) must equal a CPU inverse, and the
/// forward/inverse pair must round-trip. Before the fix, `lagrange_to_coeff` returned
/// (1/n)*forward-NTT instead of the inverse.
#[test]
fn gpu_inverse_ntt_matches_cpu_and_roundtrips() {
    if !gpu_present() {
        return;
    }
    for k in [12u32, 14, 16] {
        let n = 1usize << k;
        let domain = EvaluationDomain::<Fq>::new(1, k);
        let mut rng = ChaCha8Rng::seed_from_u64(100 + k as u64);
        let evals: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();

        // GPU inverse: lagrange_to_coeff -> Self::ifft -> GPU inverse NTT.
        let gpu_coeff = domain.lagrange_to_coeff(domain.lagrange_from_vec(evals.clone()));

        // CPU reference inverse: best_fft(omega_inv) then scale by 1/n.
        let mut cpu = evals.clone();
        let omega_inv = omega_for(k).invert().unwrap();
        midnight_curves::fft::best_fft(&mut cpu, omega_inv, k);
        let n_inv = Fq::from(n as u64).invert().unwrap();
        for v in cpu.iter_mut() {
            *v *= n_inv;
        }
        assert_eq!(&*gpu_coeff, cpu.as_slice(), "GPU inverse NTT != CPU at k={k}");

        // Round-trip: forward(inverse(x)) == x.
        let back = domain.coeff_to_lagrange(gpu_coeff);
        assert_eq!(&*back, evals.as_slice(), "GPU ifft/fft round-trip failed at k={k}");
    }
}

/// Overall commit correctness: a GPU KZG Lagrange commitment must equal the BLST result.
/// Uses k >= 15 so the commit MSM actually runs on the GPU.
#[test]
fn gpu_commit_lagrange_matches_blst() {
    if !gpu_present() {
        return;
    }
    for k in [15u32, 16] {
        let mut rng = ChaCha8Rng::seed_from_u64(7 + k as u64);
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
        let domain = EvaluationDomain::<Fq>::new(1, k);

        let mut lag = domain.empty_lagrange();
        for v in lag.iter_mut() {
            *v = Fq::random(&mut rng);
        }

        // GPU commit (cached-bases MSM at k >= 15).
        let gpu_c = KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &lag);

        // BLST reference: direct multi_exp over g_lagrange.
        let scalars: Vec<Fq> = lag.iter().copied().collect();
        let cpu_c = G1Projective::multi_exp(params.g_lagrange(), &scalars);

        assert_eq!(gpu_c, cpu_c, "GPU commit_lagrange != BLST at k={k}");
    }
}

/// C2 canary: after `downsize`, the GPU base cache must be invalidated so commitments use
/// the downsized bases (not the stale larger-domain cache).
#[test]
fn gpu_downsize_invalidates_base_cache() {
    if !gpu_present() {
        return;
    }
    let (k_big, k_small) = (16u32, 15u32);
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let mut params = ParamsKZG::<Bls12>::unsafe_setup(k_big, &mut rng);

    // Populate the GPU Lagrange cache at the larger domain.
    let domain_big = EvaluationDomain::<Fq>::new(1, k_big);
    let mut lag_big = domain_big.empty_lagrange();
    for v in lag_big.iter_mut() {
        *v = Fq::random(&mut rng);
    }
    let _ = KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &lag_big);

    // Downsize, then commit at the smaller domain.
    params.downsize(k_small);
    let domain_small = EvaluationDomain::<Fq>::new(1, k_small);
    let mut lag_small = domain_small.empty_lagrange();
    for v in lag_small.iter_mut() {
        *v = Fq::random(&mut rng);
    }
    let gpu_c = KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &lag_small);

    let scalars: Vec<Fq> = lag_small.iter().copied().collect();
    let cpu_c = G1Projective::multi_exp(params.g_lagrange(), &scalars);
    assert_eq!(gpu_c, cpu_c, "downsized GPU commit != BLST (stale cache?)");
}

/// Batch domain transforms must equal their per-polynomial counterparts
/// element-wise (they share the GPU batch path, which the prover doesn't use
/// yet — this keeps the API honest until device residency makes it pay).
#[test]
fn gpu_domain_batch_transforms_match_singles() {
    if !gpu_present() {
        return;
    }
    let k = 13u32;
    let n = 1usize << k;
    let b = 4usize;
    let domain = EvaluationDomain::<Fq>::new(2, k);
    let mut rng = ChaCha8Rng::seed_from_u64(99);

    let lagrange_polys: Vec<_> = (0..b)
        .map(|_| {
            let mut p = domain.empty_lagrange();
            for v in p.iter_mut() {
                *v = Fq::random(&mut rng);
            }
            p
        })
        .collect();

    let singles: Vec<_> = lagrange_polys
        .iter()
        .map(|p| domain.lagrange_to_coeff(p.clone()))
        .collect();
    let batched = domain.lagrange_to_coeff_batch(lagrange_polys);
    for (i, (s, bt)) in singles.iter().zip(batched.iter()).enumerate() {
        assert_eq!(s.len(), n);
        assert!(
            s.iter().zip(bt.iter()).all(|(a, b)| a == b),
            "lagrange_to_coeff_batch differs from singles at poly {i}"
        );
    }

    let ext_singles: Vec<_> = singles
        .iter()
        .map(|p| domain.coeff_to_extended(p.clone()))
        .collect();
    let ext_batched = domain.coeff_to_extended_batch(&singles);
    for (i, (s, bt)) in ext_singles.iter().zip(ext_batched.iter()).enumerate() {
        assert!(
            s.iter().zip(bt.iter()).all(|(a, b)| a == b) && s.len() == bt.len(),
            "coeff_to_extended_batch differs from singles at poly {i}"
        );
    }
    println!("✓ domain batch transforms == singles (b={b} × 2^{k})");
}

// =============================================================================
// End-to-end: a full GPU-accelerated proof must VERIFY (the ultimate GATE-0 gate).
// =============================================================================

mod e2e {
    use super::*;
    use blake2b_simd::State as Blake2bState;
    use midnight_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            create_proof as create_plonk_proof, keygen_pk, keygen_vk,
            prepare as prepare_plonk_proof, Advice, Circuit, Column, Constraints,
            ConstraintSystem, Error, Selector,
        },
        poly::{commitment::Guard, Rotation},
        transcript::{CircuitTranscript, Transcript},
    };

    /// Domain-filling circuit (same shape as `e2e_proof_benchmark`): 8 advice columns,
    /// fills half the domain so polynomial sizes equal the domain size and the GPU NTT/MSM
    /// paths are actually exercised.
    #[derive(Clone)]
    struct DomainFillingCircuit {
        k: u32,
        fill_ratio: f64,
    }

    #[derive(Clone, Debug)]
    struct DomainFillingConfig {
        advice_cols: Vec<Column<Advice>>,
        selector: Selector,
    }

    impl Circuit<Fq> for DomainFillingCircuit {
        type Config = DomainFillingConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fq>) -> Self::Config {
            let advice_cols: Vec<_> = (0..8).map(|_| meta.advice_column()).collect();
            let selector = meta.selector();
            meta.create_gate("domain_fill", |vc| {
                let a0 = vc.query_advice(advice_cols[0], Rotation::cur());
                Constraints::with_selector(selector, vec![a0.clone() - a0])
            });
            DomainFillingConfig { advice_cols, selector }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fq>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "domain_fill_region",
                |mut region| {
                    let domain_size = 1usize << self.k;
                    let rows_to_fill = ((domain_size as f64) * self.fill_ratio) as usize;
                    for row in 0..rows_to_fill {
                        config.selector.enable(&mut region, row)?;
                        for (col_idx, col) in config.advice_cols.iter().enumerate() {
                            region.assign_advice(
                                || format!("a[{col_idx}][{row}]"),
                                *col,
                                row,
                                || Value::known(Fq::from((row + col_idx + 1) as u64)),
                            )?;
                        }
                    }
                    Ok(())
                },
            )
        }
    }

    fn prove_and_verify(k: u32) {
        let mut rng = ChaCha8Rng::seed_from_u64(42 + k as u64);
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
        let circuit = DomainFillingCircuit { k, fill_ratio: 0.5 };

        let vk = keygen_vk::<_, KZGCommitmentScheme<Bls12>, _>(&params, &circuit)
            .expect("keygen_vk");
        let pk = keygen_pk(vk, &circuit).expect("keygen_pk");

        let mut transcript = CircuitTranscript::<Blake2bState>::init();
        create_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _, _>(
            &params,
            &pk,
            &[circuit],
            #[cfg(feature = "committed-instances")]
            0,
            &[&[]],
            &mut transcript,
            &mut rng,
        )
        .expect("create_proof");
        let proof = transcript.finalize();

        // The step the benchmark skipped: actually verify.
        let mut vt = CircuitTranscript::<Blake2bState>::init_from_bytes(&proof);
        let guard = prepare_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _>(
            pk.get_vk(),
            #[cfg(feature = "committed-instances")]
            &[&[]],
            &[&[]],
            &mut vt,
        )
        .expect("prepare");
        assert!(
            guard.verify(&params.verifier_params()).is_ok(),
            "GPU-generated proof failed to verify at k={k}"
        );
    }

    /// K=12 exercises the GPU NTT (threshold 2^12); K=16 exercises GPU NTT + GPU MSM.
    #[test]
    fn gpu_full_proof_verifies() {
        if !gpu_present() {
            return;
        }
        for k in [12u32, 14, 16] {
            prove_and_verify(k);
        }
    }

    /// K=18 is memory/time heavy; run explicitly with `-- --ignored`.
    #[test]
    #[ignore = "heavy (K=18); run with --ignored on a machine with enough VRAM"]
    fn gpu_full_proof_verifies_large() {
        if !gpu_present() {
            return;
        }
        prove_and_verify(18);
    }

    /// Proof-level timing: keygen / prove / verify per K, proof always verified.
    /// Deliberately NOT gated on `gpu_present()` — pointing `ICICLE_BACKEND_INSTALL_DIR`
    /// at a nonexistent dir is the pure-CPU arm of the comparison.
    ///
    ///   E2E_BENCH_KS=12,14,16 ICICLE_BACKEND_INSTALL_DIR=... \
    ///     cargo test --test gpu_correctness --features gpu --release -- \
    ///     e2e::gpu_e2e_proof_benchmark --ignored --nocapture
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn gpu_e2e_proof_benchmark() {
        use std::time::Instant;

        let ks: Vec<u32> = std::env::var("E2E_BENCH_KS")
            .map(|s| {
                s.split(',')
                    .map(|t| t.trim().parse().expect("bad E2E_BENCH_KS"))
                    .collect()
            })
            .unwrap_or_else(|_| vec![12, 14, 16]);

        println!(
            "\ngpu={} backend_dir={}",
            midnight_bls12_381_cuda::is_gpu_available(),
            std::env::var("ICICLE_BACKEND_INSTALL_DIR").unwrap_or_else(|_| "<default>".into()),
        );
        println!(
            "{:>4} {:>12} {:>12} {:>12}",
            "K", "keygen(ms)", "prove(ms)", "verify(ms)"
        );

        for k in ks {
            let mut rng = ChaCha8Rng::seed_from_u64(42 + k as u64);
            let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
            let circuit = DomainFillingCircuit { k, fill_ratio: 0.5 };

            let t = Instant::now();
            let vk = keygen_vk::<_, KZGCommitmentScheme<Bls12>, _>(&params, &circuit)
                .expect("keygen_vk");
            let pk = keygen_pk(vk, &circuit).expect("keygen_pk");
            let keygen_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let mut transcript = CircuitTranscript::<Blake2bState>::init();
            create_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _, _>(
                &params,
                &pk,
                &[circuit],
                #[cfg(feature = "committed-instances")]
                0,
                &[&[]],
                &mut transcript,
                &mut rng,
            )
            .expect("create_proof");
            let proof = transcript.finalize();
            let prove_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let mut vt = CircuitTranscript::<Blake2bState>::init_from_bytes(&proof);
            let guard = prepare_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _>(
                pk.get_vk(),
                #[cfg(feature = "committed-instances")]
                &[&[]],
                &[&[]],
                &mut vt,
            )
            .expect("prepare");
            assert!(
                guard.verify(&params.verifier_params()).is_ok(),
                "proof failed to verify at k={k}"
            );
            let verify_ms = t.elapsed().as_secs_f64() * 1e3;

            println!("{k:>4} {keygen_ms:>12.1} {prove_ms:>12.1} {verify_ms:>12.1}");
        }
    }

    /// Trust boundary: a proof generated with the GPU backend must be accepted by the
    /// pure-CPU BLST verifier.
    ///
    /// - PROVE (GPU): at k=16 the prover runs GPU NTT (threshold 2^12) and GPU MSM
    ///   (threshold 2^13); `should_use_gpu(2^16)` is asserted so this really is the
    ///   GPU path, not a silent CPU fallback.
    /// - VERIFY (CPU/BLST): the KZG verifier's pairing-check accumulator
    ///   (`DualMSM::eval`) calls `msm_specific` unconditionally — there is no
    ///   `should_use_gpu` gate on it — so `guard.verify` is a genuine CPU/BLST check
    ///   of the GPU-produced proof. `is_blst_available::<G1Affine>()` is asserted up
    ///   front (msm_specific panics without it).
    ///
    /// A one-bit tamper of the proof must be rejected, so the positive check is not
    /// vacuous.
    #[test]
    fn gpu_proof_verifies_on_cpu_blst() {
        use midnight_curves::G1Affine;
        use midnight_proofs::poly::kzg::msm::is_blst_available;

        if !gpu_present() {
            return;
        }
        assert!(
            is_blst_available::<G1Affine>(),
            "BLST must be available for the CPU verifier"
        );

        let k = 16u32;
        assert!(
            midnight_bls12_381_cuda::should_use_gpu(1usize << k),
            "k={k} must be above the GPU MSM threshold for this test to prove on GPU"
        );
        assert!(
            midnight_bls12_381_cuda::should_use_gpu_ntt(1usize << k),
            "k={k} must be above the GPU NTT threshold for this test to prove on GPU"
        );

        let mut rng = ChaCha8Rng::seed_from_u64(42 + k as u64);
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
        let circuit = DomainFillingCircuit { k, fill_ratio: 0.5 };

        let vk = keygen_vk::<_, KZGCommitmentScheme<Bls12>, _>(&params, &circuit)
            .expect("keygen_vk");
        let pk = keygen_pk(vk, &circuit).expect("keygen_pk");

        // The proof's GPU work must be launched on this fork's backend line
        // (the line `gpu_accel` names): its per-line MSM and NTT counters move.
        use midnight_proofs::gpu_accel::backend::{
            self,
            diag::{gpu_op_count, OpKind},
        };
        let line = backend::line_id();
        let msm_launches_before = gpu_op_count(line, OpKind::Msm);
        let ntt_launches_before = gpu_op_count(line, OpKind::Ntt);

        // ---- Generate the proof on GPU (GPU NTT + GPU MSM) ----
        let mut transcript = CircuitTranscript::<Blake2bState>::init();
        create_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _, _>(
            &params,
            &pk,
            &[circuit],
            #[cfg(feature = "committed-instances")]
            0,
            &[&[]],
            &mut transcript,
            &mut rng,
        )
        .expect("create_proof");
        let proof = transcript.finalize();

        assert!(
            gpu_op_count(line, OpKind::Msm) > msm_launches_before,
            "the k={k} proof launched no GPU MSM on backend line {line}"
        );
        assert!(
            gpu_op_count(line, OpKind::Ntt) > ntt_launches_before,
            "the k={k} proof launched no GPU NTT on backend line {line}"
        );

        // ---- Verify on CPU/BLST (DualMSM::eval -> msm_specific, no GPU gate) ----
        let verify = |bytes: &[u8]| -> bool {
            let mut vt = CircuitTranscript::<Blake2bState>::init_from_bytes(bytes);
            match prepare_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _>(
                pk.get_vk(),
                #[cfg(feature = "committed-instances")]
                &[&[]],
                &[&[]],
                &mut vt,
            ) {
                Ok(guard) => guard.verify(&params.verifier_params()).is_ok(),
                Err(_) => false,
            }
        };

        assert!(
            verify(&proof),
            "GPU-generated proof failed CPU/BLST verification at k={k}"
        );

        // Negative control: a corrupted proof must be rejected.
        let mut tampered = proof.clone();
        tampered[0] ^= 0x01;
        assert!(
            !verify(&tampered),
            "tampered proof was accepted by the CPU/BLST verifier"
        );
    }
}
