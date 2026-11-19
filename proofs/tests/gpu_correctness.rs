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

use common::{
    blst_commit_lagrange, expect_gpu_ops, gpu_msm_k, gpu_ntt_k, gpu_test, omega_for,
    tamper_first_commitment, Verdict,
};
use ff::Field;
use midnight_curves::{Bls12, Fq};
use midnight_proofs::gpu_accel::backend::diag::OpKind;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use midnight_proofs::poly::{
    commitment::PolynomialCommitmentScheme,
    kzg::{params::ParamsKZG, KZGCommitmentScheme},
    EvaluationDomain,
};

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

        assert_eq!(gpu_c, blst_commit_lagrange(&params, &lag), "GPU commit_lagrange != BLST at k={k}");
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

    assert_eq!(gpu_c, blst_commit_lagrange(&params, &lag_small), "downsized GPU commit != BLST (stale cache?)");
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

    /// Proves the circuit at `k` with GPU MSMs and NTTs on this fork's backend
    /// line and no fallback, then checks the proof with BLST's pairing check
    /// (`common::Verdict`, a verifier that does not touch the GPU) and with the
    /// verifier itself. Both must reject the same proof with its first
    /// commitment moved to another valid point: it still parses, so the
    /// rejection comes from the pairing. The verifier's own verdict on a bad
    /// proof is asserted nowhere else: a final check that passes whatever it
    /// is given fails only this assertion.
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

        let prepare = |bytes: &[u8]| {
            let mut vt = CircuitTranscript::<Blake2bState>::init_from_bytes(bytes);
            prepare_plonk_proof::<_, KZGCommitmentScheme<Bls12>, _>(
                pk.get_vk(),
                #[cfg(feature = "committed-instances")]
                &[&[]],
                &[&[]],
                &mut vt,
            )
        };
        // The verifier's own check: its MSMs are routed by size, so on the GPU
        // under forced GPU mode.
        let verifies = |bytes: &[u8]| {
            prepare(bytes).expect("the proof parses").verify(&params.verifier_params()).is_ok()
        };
        let tampered = tamper_first_commitment(&proof);
        assert_eq!(Verdict::of(&params, prepare(&proof)), Verdict::Accepted, "GPU-generated proof, k={k}");
        assert_eq!(
            Verdict::of(&params, prepare(&tampered)),
            Verdict::Rejected,
            "a proof with a moved commitment, k={k}"
        );
        assert!(verifies(&proof), "the verifier rejects the GPU-generated proof, k={k}");
        assert!(!verifies(&tampered), "the verifier accepts a proof with a moved commitment, k={k}");
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
}
