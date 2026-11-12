//! GPU proof fidelity: a GPU-built prover must produce the same proof BYTES as the CPU
//! prover, not merely a proof that verifies.
//!
//! The GPU delta reorders work in the prover: lookup products, permutation products and
//! trashcans are computed first and then batch-committed (`commit_lagrange_batch`), and
//! only afterwards written to the transcript. That is correct only if every random draw
//! and every transcript write still happens in the upstream order and every commitment
//! is the same group element. `gpu_correctness` checks that GPU proofs verify; this test
//! checks they are identical to CPU proofs, on a circuit that exercises every batched
//! path at GPU sizes:
//!   * two circuit instances in one proof (cross-circuit flattening of the batches);
//!   * two lookups per circuit (lookup-product batch of 4);
//!   * copy constraints over 5 equality columns, incl. both instance columns
//!     (2 permutation product polynomials per circuit);
//!   * two additive-selector gates (2 trashcans per circuit, trash batch of 4);
//!   * a second advice phase with 2 columns that depend on a challenge;
//!   * committed instance columns;
//!   * k = 14 (above the 2^13 GPU MSM threshold).
//!
//! Both instance columns are committed: with several circuits per proof, midnight-proofs
//! 0.8.2 absorbs committed and plain instances in a different order in the prover
//! (per circuit) and the verifier (all commitments first), so a proof mixing the two
//! kinds does not verify. That is upstream behaviour, unrelated to the GPU.
//!
//! The prover rng is a seeded ChaCha and the lookups leave a single leftover table value
//! (see `row_values`), so a proof is a deterministic function of the code path. Every
//! run, whatever the device, asserts that the proof's blake2b-128 is `EXPECTED_DIGEST`:
//! the bytes upstream's CPU prover (midnight-proofs 0.8.2) emits for this circuit and
//! these seeds. The bytes also depend on the versions of the dependencies (this workspace
//! does not pin them in git): after a dependency update, or if an upstream prover change
//! legitimately changes them, check that pristine upstream 0.8.2 built with the same
//! `Cargo.lock` prints the same value (this file runs there once its `gpu` gate is
//! dropped), then regenerate the constant from a `MIDNIGHT_DEVICE=cpu` run, which prints
//! it.
//!
//! To locate a difference, compare CPU and GPU byte by byte across two processes (the
//! device is fixed per process: `MIDNIGHT_DEVICE` is read once):
//!
//!   MIDNIGHT_DEVICE=cpu FIDELITY_DUMP=/tmp/cpu.bin \
//!     cargo test --release --features gpu --test gpu_proof_fidelity -- --nocapture
//!   MIDNIGHT_DEVICE=gpu MIDNIGHT_GPU_STRICT=1 FIDELITY_EXPECT=/tmp/cpu.bin \
//!     cargo test --release --features gpu --test gpu_proof_fidelity -- --nocapture
//!
//! `FIDELITY_DUMP` writes the proof bytes; `FIDELITY_EXPECT` asserts byte equality with a
//! previous dump. The proof is always verified with an explicit CPU (BLST) pairing check.
//! Byte equality also covers the verifying key: the transcript starts with the vk hash,
//! so different fixed commitments would change every challenge.

#![cfg(all(feature = "gpu", feature = "committed-instances"))]

use blake2b_simd::State as Blake2bState;
use ff::Field;
use group::{Curve, Group};
use midnight_curves::{
    pairing::{MillerLoopResult, MultiMillerLoop},
    Bls12, Fq, G1Projective, G2Prepared,
};
use midnight_proofs::{
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    plonk::{
        commit_to_instances, create_proof, keygen_pk, keygen_vk_with_k, prepare, Advice, Challenge,
        Circuit, Column, ConstraintSystem, Constraints, Error, FirstPhase, Instance, SecondPhase,
        Selector, TableColumn,
    },
    poly::{
        commitment::Guard,
        kzg::{msm::DualMSM, params::ParamsKZG, KZGCommitmentScheme},
        Rotation,
    },
    transcript::{CircuitTranscript, Transcript},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Scheme = KZGCommitmentScheme<Bls12>;

const K: u32 = 14;
/// Rows carrying gates (half the domain).
const ROWS: usize = 1 << (K - 1);
/// Lookup table: 0..TABLE_SIZE.
const TABLE_SIZE: u64 = 256;

/// blake2b-128 of the proof as upstream's CPU prover emits it (see the module docs).
/// `truncated-challenges` changes every challenge, so it does not apply to it.
#[cfg(not(feature = "truncated-challenges"))]
const EXPECTED_DIGEST: &str = "36bb0fcf18a23ef26c73e47141635f2d";

#[derive(Clone, Debug)]
struct FidelityConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    d: Column<Advice>,
    e: Column<Advice>,
    inputs: Column<Instance>,
    outputs: Column<Instance>,
    q_mul: Selector,
    q_add: Selector,
    q_sq: Selector,
    q_rlc: Selector,
    q_lookup: Selector,
    table: TableColumn,
    challenge: Challenge,
}

/// First row whose `a` is copied from the previous row's `b`.
const FIRST_COPY_ROW: usize = 2 * TABLE_SIZE as usize;

/// Row values of the circuit with the given seed: (a, b, c).
///
/// Rows `0..TABLE_SIZE` take every table value in both `a` and `b` (7 and 13 are odd),
/// so each lookup leaves only zeros over in the table. That keeps the proof
/// reproducible: the lookup prover fills repeated rows from a `HashMap` of leftover
/// table values, and with two or more distinct leftovers its per-process random
/// iteration order changes the permuted table column, and so the proof bytes, from one
/// run to the next (CPU or GPU alike).
fn row_values(seed: u64) -> Vec<(u64, u64, u64)> {
    let mut rows: Vec<(u64, u64, u64)> = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let b = (3 * seed + 13 * i as u64 + 5) % TABLE_SIZE;
        // Every 16th row from FIRST_COPY_ROW on copies `a` from the previous row's `b`
        // (copy constraint).
        let a = if i >= FIRST_COPY_ROW && i % 16 == 1 {
            rows[i - 1].1
        } else {
            (seed + 7 * i as u64) % TABLE_SIZE
        };
        let c = if i % 2 == 0 { a * b } else { a + b };
        rows.push((a, b, c));
    }
    rows
}

#[derive(Clone)]
struct FidelityCircuit {
    seed: u64,
}

impl FidelityCircuit {
    /// The two (committed) instance columns: (a_0, b_0) and (c_0, c_1).
    fn instances(&self) -> [Vec<Fq>; 2] {
        let rows = row_values(self.seed);
        [
            vec![Fq::from(rows[0].0), Fq::from(rows[0].1)],
            vec![Fq::from(rows[0].2), Fq::from(rows[1].2)],
        ]
    }
}

impl Circuit<Fq> for FidelityCircuit {
    type Config = FidelityConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fq>) -> Self::Config {
        let a = meta.advice_column_in(FirstPhase);
        let b = meta.advice_column_in(FirstPhase);
        let c = meta.advice_column_in(FirstPhase);
        let challenge = meta.challenge_usable_after(FirstPhase);
        let d = meta.advice_column_in(SecondPhase);
        let e = meta.advice_column_in(SecondPhase);
        let inputs = meta.instance_column();
        let outputs = meta.instance_column();
        let table = meta.lookup_table_column();

        for column in [a, b, c] {
            meta.enable_equality(column);
        }
        meta.enable_equality(inputs);
        meta.enable_equality(outputs);

        let q_mul = meta.selector();
        let q_add = meta.complex_selector();
        let q_sq = meta.complex_selector();
        let q_rlc = meta.selector();
        let q_lookup = meta.complex_selector();

        meta.create_gate("mul", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            Constraints::with_selector(q_mul, vec![a * b - c])
        });

        // Two additive-selector gates: two trashcans.
        meta.create_gate("add (trashcan)", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            Constraints::with_additive_selector(q_add, vec![a + b - c])
        });
        meta.create_gate("mul (trashcan)", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            Constraints::with_additive_selector(q_sq, vec![c - a * b])
        });

        // Second phase: d = a + r*b + r^2*c, e = d^2.
        meta.create_gate("rlc", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let d = meta.query_advice(d, Rotation::cur());
            let e = meta.query_advice(e, Rotation::cur());
            let r = meta.query_challenge(challenge);
            Constraints::with_selector(
                q_rlc,
                vec![
                    d.clone() - (a + r.clone() * b + r.clone() * r * c),
                    e - d.clone() * d,
                ],
            )
        });

        meta.lookup("a in table", |meta| {
            let q = meta.query_selector(q_lookup);
            let a = meta.query_advice(a, Rotation::cur());
            vec![(q * a, table)]
        });
        meta.lookup("b in table", |meta| {
            let q = meta.query_selector(q_lookup);
            let b = meta.query_advice(b, Rotation::cur());
            vec![(q * b, table)]
        });

        FidelityConfig {
            a,
            b,
            c,
            d,
            e,
            inputs,
            outputs,
            q_mul,
            q_add,
            q_sq,
            q_rlc,
            q_lookup,
            table,
            challenge,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fq>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "range table",
            |mut table| {
                for v in 0..TABLE_SIZE {
                    table.assign_cell(
                        || "table",
                        config.table,
                        v as usize,
                        || Value::known(Fq::from(v)),
                    )?;
                }
                Ok(())
            },
        )?;

        let r = layouter.get_challenge(config.challenge);
        let rows = row_values(self.seed);

        let (first_a, first_b, first_c, second_c) = layouter.assign_region(
            || "main",
            |mut region| {
                let mut first = None;
                let mut second_c = None;
                let mut prev_b: Option<AssignedCell<Fq, Fq>> = None;
                for (i, &(a, b, c)) in rows.iter().enumerate() {
                    let (a, b, c) = (Fq::from(a), Fq::from(b), Fq::from(c));
                    config.q_rlc.enable(&mut region, i)?;
                    config.q_lookup.enable(&mut region, i)?;
                    if i % 2 == 0 {
                        config.q_mul.enable(&mut region, i)?;
                    } else {
                        config.q_add.enable(&mut region, i)?;
                    }
                    if i % 4 == 0 {
                        config.q_sq.enable(&mut region, i)?;
                    }

                    let a_cell = region.assign_advice(|| "a", config.a, i, || Value::known(a))?;
                    let b_cell = region.assign_advice(|| "b", config.b, i, || Value::known(b))?;
                    let c_cell = region.assign_advice(|| "c", config.c, i, || Value::known(c))?;
                    let d = r.map(|r| a + r * b + r * r * c);
                    region.assign_advice(|| "d", config.d, i, || d)?;
                    region.assign_advice(|| "e", config.e, i, || d.map(|d| d * d))?;

                    if i >= FIRST_COPY_ROW && i % 16 == 1 {
                        region.constrain_equal(prev_b.as_ref().unwrap().cell(), a_cell.cell())?;
                    }
                    if i == 0 {
                        first = Some((a_cell.clone(), b_cell.clone(), c_cell.clone()));
                    }
                    if i == 1 {
                        second_c = Some(c_cell.clone());
                    }
                    prev_b = Some(b_cell);
                }
                let (a, b, c) = first.unwrap();
                Ok((a, b, c, second_c.unwrap()))
            },
        )?;

        layouter.constrain_instance(first_a.cell(), config.inputs, 0)?;
        layouter.constrain_instance(first_b.cell(), config.inputs, 1)?;
        layouter.constrain_instance(first_c.cell(), config.outputs, 0)?;
        layouter.constrain_instance(second_c.cell(), config.outputs, 1)?;
        Ok(())
    }
}

/// Pure CPU (BLST) final check of a KZG verification guard: the same pairing equation as
/// `DualMSM::check`, but with `G1Projective::multi_exp` instead of `msm_specific`, which
/// would run on the GPU under `MIDNIGHT_DEVICE=gpu`.
fn verify_on_cpu(params: &ParamsKZG<Bls12>, guard: &DualMSM<Bls12>) -> bool {
    fn cpu_msm<L>(terms: &[(&L, &Fq, &G1Projective)]) -> G1Projective {
        let scalars: Vec<Fq> = terms.iter().map(|t| *t.1).collect();
        let bases: Vec<G1Projective> = terms.iter().map(|t| *t.2).collect();
        G1Projective::multi_exp(&bases, &scalars)
    }
    let (left, right) = guard.split();
    let (left, right) = (cpu_msm(&left).to_affine(), cpu_msm(&right).to_affine());
    let s_g2 = G2Prepared::from(params.s_g2().to_affine());
    let n_g2 = G2Prepared::from((-params.g2()).to_affine());
    bool::from(
        Bls12::multi_miller_loop(&[(&left, &s_g2), (&right, &n_g2)])
            .final_exponentiation()
            .is_identity(),
    )
}

/// Commitment to a committed-instance column, computed with a CPU (BLST) MSM.
fn cpu_instance_commitment(params: &ParamsKZG<Bls12>, values: &[Fq]) -> G1Projective {
    let mut scalars = vec![Fq::ZERO; 1 << K];
    scalars[..values.len()].copy_from_slice(values);
    G1Projective::multi_exp(params.g_lagrange(), &scalars)
}

#[test]
fn gpu_proof_bytes_match_cpu() {
    let device = std::env::var("MIDNIGHT_DEVICE").unwrap_or_else(|_| "auto".into());
    #[cfg(feature = "gpu")]
    let gpu_msm = {
        let gpu_msm = midnight_bls12_381_cuda::should_use_gpu(1 << K);
        if device.eq_ignore_ascii_case("gpu") {
            assert!(
                midnight_bls12_381_cuda::is_gpu_available(),
                "MIDNIGHT_DEVICE=gpu but no GPU"
            );
        }
        if device.eq_ignore_ascii_case("cpu") {
            assert!(
                !gpu_msm,
                "MIDNIGHT_DEVICE=cpu must keep every MSM on the CPU"
            );
        }
        midnight_bls12_381_cuda::config::reset_gpu_fallback_count();
        gpu_msm
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_msm = false;

    let params = ParamsKZG::<Bls12>::unsafe_setup(K, ChaCha8Rng::seed_from_u64(0xF1DE));
    let circuits = [FidelityCircuit { seed: 1 }, FidelityCircuit { seed: 2 }];

    // `keygen_vk` sizes the circuit with the cost model, which panics on selectors
    // enabled during the second-phase synthesis pass; the size is known here.
    let vk = keygen_vk_with_k::<_, Scheme, _>(&params, &circuits[0], K).expect("keygen_vk");
    let pk = keygen_pk(vk, &circuits[0]).expect("keygen_pk");

    let instances: Vec<[Vec<Fq>; 2]> = circuits.iter().map(|c| c.instances()).collect();
    let prover_instances: Vec<[&[Fq]; 2]> =
        instances.iter().map(|[x, y]| [&x[..], &y[..]]).collect();
    let prover_instances: Vec<&[&[Fq]]> = prover_instances.iter().map(|i| &i[..]).collect();

    let mut transcript = CircuitTranscript::<Blake2bState>::init();
    create_proof::<_, Scheme, _, _>(
        &params,
        &pk,
        &circuits,
        2,
        &prover_instances,
        &mut transcript,
        ChaCha8Rng::seed_from_u64(0x5EED),
    )
    .expect("create_proof");
    let proof = transcript.finalize();

    #[cfg(feature = "gpu")]
    let gpu_fallbacks = midnight_bls12_381_cuda::config::gpu_fallback_count();
    #[cfg(not(feature = "gpu"))]
    let gpu_fallbacks = 0;
    let digest = blake2b_simd::Params::new().hash_length(16).hash(&proof);
    println!(
        "MIDNIGHT_DEVICE={device} gpu_msm(2^{K})={gpu_msm} gpu_fallbacks={gpu_fallbacks} \
         proof_len={} blake2b-128={}",
        proof.len(),
        digest.to_hex(),
    );

    // Instance commitments: CPU (BLST) MSM, cross-checked against the prover-side helper
    // (which runs on the GPU under MIDNIGHT_DEVICE=gpu).
    let committed: Vec<Vec<G1Projective>> = instances
        .iter()
        .map(|columns| {
            columns
                .iter()
                .map(|values| {
                    let cpu = cpu_instance_commitment(&params, values);
                    let domain = pk.get_vk().get_domain();
                    let helper = commit_to_instances::<Fq, Scheme>(&params, domain, values);
                    assert_eq!(cpu, helper, "commit_to_instances differs from the CPU MSM");
                    cpu
                })
                .collect()
        })
        .collect();
    let committed: Vec<&[G1Projective]> = committed.iter().map(|c| &c[..]).collect();
    let no_plain_instances: Vec<&[&[Fq]]> = vec![&[]; circuits.len()];

    let prepare_guard = |bytes: &[u8]| -> Option<DualMSM<Bls12>> {
        let mut vt = CircuitTranscript::<Blake2bState>::init_from_bytes(bytes);
        prepare::<_, Scheme, _>(pk.get_vk(), &committed, &no_plain_instances, &mut vt).ok()
    };

    let guard = prepare_guard(&proof).expect("prepare");
    assert!(
        verify_on_cpu(&params, &guard),
        "proof failed the CPU (BLST) pairing check"
    );
    assert!(
        guard.verify(&params.verifier_params()).is_ok(),
        "proof failed verification"
    );

    // Negative control: a corrupted proof must be rejected by the CPU check.
    let mut tampered = proof.clone();
    let mid = tampered.len() / 2;
    tampered[mid] ^= 0x01;
    assert!(
        !prepare_guard(&tampered).is_some_and(|g| verify_on_cpu(&params, &g)),
        "tampered proof was accepted"
    );

    if let Ok(path) = std::env::var("FIDELITY_DUMP") {
        std::fs::write(&path, &proof).expect("write FIDELITY_DUMP");
        println!("proof written to {path}");
    }
    if let Ok(path) = std::env::var("FIDELITY_EXPECT") {
        let expected = std::fs::read(&path).expect("read FIDELITY_EXPECT");
        if proof != expected {
            let first_diff = proof.iter().zip(&expected).position(|(x, y)| x != y);
            panic!(
                "proof bytes differ from {path}: {} vs {} bytes, first differing byte at {:?}",
                proof.len(),
                expected.len(),
                first_diff
            );
        }
        println!("proof bytes identical to {path}");
    }
    #[cfg(not(feature = "truncated-challenges"))]
    assert_eq!(
        digest.to_hex().as_str(),
        EXPECTED_DIGEST,
        "the proof is not the upstream CPU prover's; if the upstream prover changed, update \
         EXPECTED_DIGEST"
    );

    // A forced-GPU run must really have run on the GPU. (Under MIDNIGHT_GPU_STRICT a
    // fallback already panicked; without it, the bytes above must still match.)
    #[cfg(feature = "gpu")]
    if device.eq_ignore_ascii_case("gpu") {
        assert_eq!(
            midnight_bls12_381_cuda::config::gpu_fallback_count(),
            0,
            "a GPU operation failed and fell back to CPU"
        );
    }
}
