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
//!
//! A benchmark that created proofs but never VERIFIED them left both bugs invisible;
//! `gpu_full_proof_verifies` below closes that gap.
//!
//! Run against ours vs the closed backend via the `ICICLE_BACKEND_INSTALL_DIR` A/B switch:
//!   ICICLE_BACKEND_INSTALL_DIR=~/.local/icicle-backends/ours \
//!     cargo test --test gpu_correctness --features gpu --release -- --nocapture

#![cfg(feature = "gpu")]

mod common;

use common::{expect_gpu_ops, gpu_msm_k, gpu_ntt_k, gpu_test, tamper_first_commitment, Verdict};
use ff::{Field, PrimeField};
use midnight_curves::{Bls12, Fq, G1Projective};
use midnight_proofs::gpu_accel::backend::diag::OpKind;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use midnight_proofs::poly::{
    commitment::PolynomialCommitmentScheme,
    kzg::{params::ParamsKZG, KZGCommitmentScheme},
    EvaluationDomain,
};

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
    let Some(_gpu) = gpu_test() else { return };
    let k0 = gpu_ntt_k(12);
    for k in [k0, k0 + 2, k0 + 4] {
        let n = 1usize << k;
        let domain = EvaluationDomain::<Fq>::new(1, k);
        let mut rng = ChaCha8Rng::seed_from_u64(k as u64);
        let coeffs: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();

        // GPU path: coeff_to_lagrange runs a forward NTT through the GPU dispatcher.
        let gpu = expect_gpu_ops(&[OpKind::Ntt], || {
            domain.coeff_to_lagrange(domain.coeff_from_vec(coeffs.clone()))
        });

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
    let Some(_gpu) = gpu_test() else { return };
    let k0 = gpu_ntt_k(12);
    for k in [k0, k0 + 2, k0 + 4] {
        let n = 1usize << k;
        let domain = EvaluationDomain::<Fq>::new(1, k);
        let mut rng = ChaCha8Rng::seed_from_u64(100 + k as u64);
        let evals: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();

        // GPU inverse: lagrange_to_coeff -> Self::ifft -> GPU inverse NTT.
        let gpu_coeff = expect_gpu_ops(&[OpKind::Ntt], || {
            domain.lagrange_to_coeff(domain.lagrange_from_vec(evals.clone()))
        });

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

/// Overall commit correctness: a GPU KZG Lagrange commitment must equal the BLST result,
/// at sizes the routing sends to the GPU (the commit's cached-bases MSM).
#[test]
fn gpu_commit_lagrange_matches_blst() {
    let Some(_gpu) = gpu_test() else { return };
    let k0 = gpu_msm_k(15);
    for k in [k0, k0 + 1] {
        let mut rng = ChaCha8Rng::seed_from_u64(7 + k as u64);
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
        let domain = EvaluationDomain::<Fq>::new(1, k);

        let mut lag = domain.empty_lagrange();
        for v in lag.iter_mut() {
            *v = Fq::random(&mut rng);
        }

        // GPU commit (cached-bases MSM).
        let gpu_c = expect_gpu_ops(&[OpKind::Msm], || {
            KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &lag)
        });

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
    let Some(_gpu) = gpu_test() else { return };
    let k_small = gpu_msm_k(15);
    let k_big = k_small + 1;
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let mut params = ParamsKZG::<Bls12>::unsafe_setup(k_big, &mut rng);

    // Populate the GPU Lagrange cache at the larger domain.
    let domain_big = EvaluationDomain::<Fq>::new(1, k_big);
    let mut lag_big = domain_big.empty_lagrange();
    for v in lag_big.iter_mut() {
        *v = Fq::random(&mut rng);
    }
    expect_gpu_ops(&[OpKind::Msm], || {
        KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &lag_big)
    });

    // Downsize, then commit at the smaller domain.
    params.downsize(k_small);
    let domain_small = EvaluationDomain::<Fq>::new(1, k_small);
    let mut lag_small = domain_small.empty_lagrange();
    for v in lag_small.iter_mut() {
        *v = Fq::random(&mut rng);
    }
    let gpu_c = expect_gpu_ops(&[OpKind::Msm], || {
        KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &lag_small)
    });

    let scalars: Vec<Fq> = lag_small.iter().copied().collect();
    let cpu_c = G1Projective::multi_exp(params.g_lagrange(), &scalars);
    assert_eq!(gpu_c, cpu_c, "downsized GPU commit != BLST (stale cache?)");
}

/// Batch domain transforms must equal their per-polynomial counterparts
/// element-wise (they share the GPU batch path, which the prover doesn't use
/// yet — this keeps the API honest until device residency makes it pay).
#[test]
fn gpu_domain_batch_transforms_match_singles() {
    let Some(_gpu) = gpu_test() else { return };
    let k = gpu_ntt_k(13);
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
    let batched = expect_gpu_ops(&[OpKind::NttBatch], || domain.lagrange_to_coeff_batch(lagrange_polys));
    assert_eq!(batched.len(), b);
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
    let ext_batched = expect_gpu_ops(&[OpKind::NttBatch], || domain.coeff_to_extended_batch(&singles));
    assert_eq!(ext_batched.len(), b);
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

    /// Domain-filling circuit: 8 advice columns over half the domain, so polynomial
    /// sizes equal the domain size and the GPU NTT/MSM paths are exercised. Its gates
    /// are not identically zero (a2 = a0 * a1, and a3 on the next row = a3 + a0) and
    /// a copy constraint ties a4 to a0 (the permutation argument), so the quotient,
    /// the extended-domain NTTs and the quotient's evaluation must be right for the
    /// proof to verify.
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

    /// The witness of column `col` at `row` (rows < `rows` are constrained).
    fn witness(col: usize, row: usize) -> Fq {
        let (a0, a1) = (Fq::from(row as u64 + 1), Fq::from(row as u64 + 2));
        match col {
            0 | 4 => a0,
            1 => a1,
            2 => a0 * a1,
            3 => Fq::from((row * (row + 1) / 2) as u64), // sum of a0 over the rows before
            _ => Fq::from((row + col + 1) as u64),
        }
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
            meta.enable_equality(advice_cols[0]);
            meta.enable_equality(advice_cols[4]);
            meta.create_gate("domain_fill", |vc| {
                let a0 = vc.query_advice(advice_cols[0], Rotation::cur());
                let a1 = vc.query_advice(advice_cols[1], Rotation::cur());
                let a2 = vc.query_advice(advice_cols[2], Rotation::cur());
                let a3 = vc.query_advice(advice_cols[3], Rotation::cur());
                let a3_next = vc.query_advice(advice_cols[3], Rotation::next());
                Constraints::with_selector(selector, vec![a2 - a0.clone() * a1, a3_next - a3 - a0])
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
                        let mut cells = Vec::with_capacity(config.advice_cols.len());
                        for (col_idx, col) in config.advice_cols.iter().enumerate() {
                            cells.push(region.assign_advice(
                                || format!("a[{col_idx}][{row}]"),
                                *col,
                                row,
                                || Value::known(witness(col_idx, row)),
                            )?);
                        }
                        if row == 0 {
                            region.constrain_equal(cells[0].cell(), cells[4].cell())?;
                        }
                    }
                    // The rotation reads a3 one row past the last constrained row.
                    region.assign_advice(
                        || "a[3] after the last row",
                        config.advice_cols[3],
                        rows_to_fill,
                        || Value::known(witness(3, rows_to_fill)),
                    )?;
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
        expect_gpu_ops(&[OpKind::Msm, OpKind::Ntt], || {
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
        })
        .expect("create_proof");
        let proof = transcript.finalize();

        // BLST's pairing check, and the verifier's own (its MSMs routed by size).
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
            common::blst_pairing_check(&params, &guard),
            "GPU-generated proof failed the BLST pairing check at k={k}"
        );
        assert!(
            guard.verify(&params.verifier_params()).is_ok(),
            "GPU-generated proof failed to verify at k={k}"
        );
    }

    /// From the smallest size both crossovers send to the GPU: every proof launches GPU
    /// MSMs and NTTs on this fork's backend line, with no fallback.
    #[test]
    fn gpu_full_proof_verifies() {
        let Some(_gpu) = gpu_test() else { return };
        let k0 = gpu_msm_k(gpu_ntt_k(12));
        for k in [k0, k0 + 2, k0 + 4] {
            prove_and_verify(k);
        }
    }

    /// K=18 is memory/time heavy; run explicitly with `-- --ignored`.
    #[test]
    #[ignore = "heavy (K=18); run with --ignored on a machine with enough VRAM"]
    fn gpu_full_proof_verifies_large() {
        let Some(_gpu) = gpu_test() else { return };
        prove_and_verify(18);
    }

    /// Proof-level timing: keygen / prove / verify per K, proof always verified.
    /// Deliberately NOT gated on `gpu_test()` — pointing `ICICLE_BACKEND_INSTALL_DIR`
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

    /// Trust boundary: a proof generated with the GPU backend must be accepted by a
    /// verifier that does not touch the GPU.
    ///
    /// - PROVE (GPU): the proof launches GPU MSMs and NTTs on this fork's backend line,
    ///   with no fallback (op counters).
    /// - VERIFY: `prepare`, then the pairing check with BLST alone
    ///   (`common::blst_pairing_check`; `guard.verify` would route its MSMs by size,
    ///   to the GPU under `MIDNIGHT_DEVICE=gpu`).
    ///
    /// The same proof with its first commitment moved to another valid point still
    /// parses and must be rejected by the pairing, so the positive check is not vacuous.
    #[test]
    fn gpu_proof_verifies_on_cpu_blst() {
        let Some(_gpu) = gpu_test() else { return };
        let k = gpu_msm_k(gpu_ntt_k(16));

        let mut rng = ChaCha8Rng::seed_from_u64(42 + k as u64);
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
        let circuit = DomainFillingCircuit { k, fill_ratio: 0.5 };

        let vk = keygen_vk::<_, KZGCommitmentScheme<Bls12>, _>(&params, &circuit)
            .expect("keygen_vk");
        let pk = keygen_pk(vk, &circuit).expect("keygen_pk");

        // ---- Generate the proof on the GPU (GPU NTT + GPU MSM on this fork's line) ----
        let mut transcript = CircuitTranscript::<Blake2bState>::init();
        expect_gpu_ops(&[OpKind::Msm, OpKind::Ntt], || {
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
        })
        .expect("create_proof");
        let proof = transcript.finalize();

        // ---- Verify: prepare, then BLST's pairing check ----
        let verdict = |bytes: &[u8]| {
            let mut vt = CircuitTranscript::<Blake2bState>::init_from_bytes(bytes);
            let prepared = prepare_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _>(
                pk.get_vk(),
                #[cfg(feature = "committed-instances")]
                &[&[]],
                &[&[]],
                &mut vt,
            );
            Verdict::of(&params, prepared)
        };
        assert_eq!(verdict(&proof), Verdict::Accepted, "GPU-generated proof, k={k}");
        assert_eq!(
            verdict(&tamper_first_commitment(&proof)),
            Verdict::Rejected,
            "a proof with a moved commitment, k={k}"
        );
    }
}
