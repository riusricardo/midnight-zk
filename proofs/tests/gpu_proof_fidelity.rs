//! GPU-vs-CPU proof fidelity: the GPU prover must emit the CPU prover's exact bytes.
//!
//! Verification alone cannot catch a GPU path that changes the proof without breaking
//! it, e.g. a reordered rng draw or transcript write in the batched lookup / trash /
//! permutation / advice commits. Proof bytes are a deterministic function of the
//! params, the circuit, the instances and the prover rng, so with a seeded rng a CPU
//! run and a GPU run must produce identical bytes.
//!
//! The circuit exercises every batched commit path at k = 14: two advice phases with a
//! challenge, two lookups, copy constraints over six equality columns, two trashcans
//! (additive selectors) and a committed instance column. Two proofs are produced:
//!   * "single": one circuit, with the first instance column in committed form;
//!   * "multi": two circuits in one proof, so the commitments that the GPU prover
//!     batches across circuits must be distributed back in order. Its instances are all
//!     plain: with committed instances, upstream's prover hashes each circuit's
//!     commitment and plain instances together, while the verifier hashes every
//!     circuit's commitments first, so a multi-circuit proof with committed instances
//!     does not verify on the CPU either.
//!
//! Each proof is verified with the final MSMs evaluated by BLST directly (never through
//! the GPU dispatch), so a GPU-produced proof is always checked on the CPU.
//!
//! Every run, on any device and with or without `gpu`, asserts that each proof's
//! blake2b-128 fingerprint is `EXPECTED_FINGERPRINTS`: the bytes upstream's CPU prover
//! (midnight-proofs 0.7.3) emits for this circuit and these seeds. The bytes also depend
//! on the versions of the dependencies (this workspace does not pin them in git): after a
//! dependency update, or if an upstream prover change legitimately changes them, check
//! that pristine upstream 0.7.3 built with the same `Cargo.lock` prints the same values,
//! then regenerate the constants from a build without `gpu` (or a `MIDNIGHT_DEVICE=cpu`
//! run), which prints them.
//!
//! To locate a difference, compare two runs byte by byte (the device is fixed per
//! process: `MIDNIGHT_DEVICE` is read once):
//!   1. `FIDELITY_DUMP=<file>` writes the proof bytes;
//!   2. `FIDELITY_EXPECT=<file>` asserts the proof bytes equal that file, reporting the
//!      first differing byte.
//!
//! ```text
//! MIDNIGHT_DEVICE=cpu FIDELITY_DUMP=/tmp/cpu.bin \
//!   cargo test -p midnight-proofs --release --features gpu --test gpu_proof_fidelity
//! MIDNIGHT_DEVICE=gpu MIDNIGHT_GPU_STRICT=1 FIDELITY_EXPECT=/tmp/cpu.bin \
//!   cargo test -p midnight-proofs --release --features gpu --test gpu_proof_fidelity
//! ```
//! The test also builds without `gpu`, which gives the upstream CPU prover's bytes.

#![cfg(feature = "committed-instances")]

use blake2b_simd::State as Blake2bState;
use group::{Curve, Group};
use midnight_curves::{
    pairing::{MillerLoopResult, MultiMillerLoop},
    Bls12, Fq, G1Projective, G2Prepared,
};
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk_with_k, prepare, Advice, Challenge, Circuit, Column,
        ConstraintSystem, Constraints, Error, FirstPhase, Instance, ProvingKey, SecondPhase,
        Selector, TableColumn, VerifyingKey,
    },
    poly::{
        kzg::{params::ParamsKZG, KZGCommitmentScheme},
        Rotation,
    },
    transcript::{CircuitTranscript, Transcript},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Scheme = KZGCommitmentScheme<Bls12>;

const K: u32 = 14;
/// Witness rows (half the domain).
const ROWS: usize = 1 << 13;
/// Rows of the committed instance column, tied to `e` by a gate.
const COMMITTED_ROWS: usize = 16;
/// Size of the lookup table (values `0..TABLE_SIZE`).
const TABLE_SIZE: u64 = 256;

const SETUP_SEED: u64 = 0x5eed_0001;
const PROVER_SEED: u64 = 0x5eed_0002;

/// blake2b-128 of each proof as upstream's CPU prover emits it (see the module docs).
/// `truncated-challenges` changes every challenge, so these do not apply to it.
#[cfg(not(feature = "truncated-challenges"))]
const EXPECTED_FINGERPRINTS: [(&str, &str); 2] = [
    ("single", "b2ba1194a0f0c76c53293b715237e23d"),
    ("multi", "004638a151665e60f2943ea1e6f11bcd"),
];

#[derive(Clone, Debug)]
struct FidelityConfig {
    // First phase.
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    e: Column<Advice>,
    // Second phase.
    d1: Column<Advice>,
    d2: Column<Advice>,
    public: Column<Instance>,
    table: TableColumn,
    q_lookup: Selector,
    q_add: Selector,
    q_mul: Selector,
    q_committed: Selector,
    q_phase2: Selector,
    challenge: Challenge,
}

/// Row `i` holds `a_i`, `b_i`, `c_i = a_i + b_i`, `e_i = a_i * b_i` (first phase) and
/// `d1_i = c_i + ch * e_i`, `d2_i = ch * d1_i + a_i` (second phase), where
/// `a_0 = x` is copied from the public instance and `a_{i+1} = b_i` is a copy
/// constraint. `a` and `b` are range-checked by lookups into `0..TABLE_SIZE`.
#[derive(Clone, Debug, Default)]
struct FidelityCircuit {
    x: u64,
    offset: u64,
}

impl FidelityCircuit {
    fn b(&self, i: usize) -> u64 {
        (i as u64 * 37 + self.offset) % TABLE_SIZE
    }

    fn a(&self, i: usize) -> u64 {
        if i == 0 {
            self.x
        } else {
            self.b(i - 1)
        }
    }

    /// Instance columns in order: committed (`e_0..e_{COMMITTED_ROWS}`), then public (`x`).
    fn instances(&self) -> [Vec<Fq>; 2] {
        let committed = (0..COMMITTED_ROWS).map(|i| Fq::from(self.a(i) * self.b(i))).collect();
        [committed, vec![Fq::from(self.x)]]
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

    fn configure(meta: &mut ConstraintSystem<Fq>) -> FidelityConfig {
        // Committed instance columns go first.
        let committed = meta.instance_column();
        let public = meta.instance_column();

        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let e = meta.advice_column();
        let challenge = meta.challenge_usable_after(FirstPhase);
        let d1 = meta.advice_column_in(SecondPhase);
        let d2 = meta.advice_column_in(SecondPhase);

        for column in [a, b, c, e, d1] {
            meta.enable_equality(column);
        }
        meta.enable_equality(public);

        let table = meta.lookup_table_column();
        let q_lookup = meta.complex_selector();
        // Additive selectors (trashcans) must be complex selectors.
        let q_add = meta.complex_selector();
        let q_mul = meta.complex_selector();
        let q_committed = meta.selector();
        let q_phase2 = meta.selector();

        for (name, column) in [("range a", a), ("range b", b)] {
            meta.lookup(name, |meta| {
                let q = meta.query_selector(q_lookup);
                let v = meta.query_advice(column, Rotation::cur());
                vec![(q * v, table)]
            });
        }

        // Two additive selectors, i.e. two trashcans.
        meta.create_gate("c = a + b", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            Constraints::with_additive_selector(q_add, vec![a + b - c])
        });
        meta.create_gate("e = a * b", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let e = meta.query_advice(e, Rotation::cur());
            Constraints::with_additive_selector(q_mul, vec![a * b - e])
        });

        meta.create_gate("committed = e", |meta| {
            let committed = meta.query_instance(committed, Rotation::cur());
            let e = meta.query_advice(e, Rotation::cur());
            Constraints::with_selector(q_committed, vec![committed - e])
        });

        meta.create_gate("second phase", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let e = meta.query_advice(e, Rotation::cur());
            let d1 = meta.query_advice(d1, Rotation::cur());
            let d2 = meta.query_advice(d2, Rotation::cur());
            let ch = meta.query_challenge(challenge);
            Constraints::with_selector(
                q_phase2,
                vec![d1.clone() - (c + ch.clone() * e), d2 - (ch * d1 + a)],
            )
        });

        FidelityConfig {
            a,
            b,
            c,
            e,
            d1,
            d2,
            public,
            table,
            q_lookup,
            q_add,
            q_mul,
            q_committed,
            q_phase2,
            challenge,
        }
    }

    fn synthesize(
        &self,
        config: FidelityConfig,
        mut layouter: impl Layouter<Fq>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "range table",
            |mut table| {
                for v in 0..TABLE_SIZE {
                    table.assign_cell(
                        || "range",
                        config.table,
                        v as usize,
                        || Value::known(Fq::from(v)),
                    )?;
                }
                Ok(())
            },
        )?;

        // Unknown in the first phase, known in the second.
        let ch = layouter.get_challenge(config.challenge);

        layouter.assign_region(
            || "main",
            |mut region| {
                region.assign_advice_from_instance(|| "a_0 = x", config.public, 0, config.a, 0)?;
                for i in 0..ROWS {
                    let (a, b) = (Fq::from(self.a(i)), Fq::from(self.b(i)));
                    let (c, e) = (a + b, a * b);

                    config.q_lookup.enable(&mut region, i)?;
                    config.q_add.enable(&mut region, i)?;
                    config.q_mul.enable(&mut region, i)?;
                    config.q_phase2.enable(&mut region, i)?;
                    if i < COMMITTED_ROWS {
                        config.q_committed.enable(&mut region, i)?;
                    }

                    let b_cell = region.assign_advice(|| "b", config.b, i, || Value::known(b))?;
                    region.assign_advice(|| "c", config.c, i, || Value::known(c))?;
                    region.assign_advice(|| "e", config.e, i, || Value::known(e))?;
                    let d1 = ch.map(|ch| c + ch * e);
                    region.assign_advice(|| "d1", config.d1, i, || d1)?;
                    region.assign_advice(
                        || "d2",
                        config.d2,
                        i,
                        || ch.zip(d1).map(|(ch, d1)| ch * d1 + a),
                    )?;

                    if i + 1 < ROWS {
                        // a_{i+1} = b_i, as a copy constraint.
                        b_cell.copy_advice(|| "a", &mut region, config.a, i + 1)?;
                    }
                }
                Ok(())
            },
        )
    }
}

/// Proves `circuits` with the first `nb_committed` instance columns in committed form.
fn prove(
    params: &ParamsKZG<Bls12>,
    pk: &ProvingKey<Fq, Scheme>,
    circuits: &[FidelityCircuit],
    nb_committed: usize,
    rng_seed: u64,
) -> Vec<u8> {
    let instances: Vec<[Vec<Fq>; 2]> = circuits.iter().map(|c| c.instances()).collect();
    let instance_refs: Vec<Vec<&[Fq]>> = instances
        .iter()
        .map(|cols| cols.iter().map(|col| col.as_slice()).collect())
        .collect();
    let instance_refs: Vec<&[&[Fq]]> = instance_refs.iter().map(|cols| cols.as_slice()).collect();

    let mut transcript = CircuitTranscript::<Blake2bState>::init();
    create_proof::<Fq, Scheme, _, _>(
        params,
        pk,
        circuits,
        nb_committed,
        &instance_refs,
        ChaCha8Rng::seed_from_u64(rng_seed),
        &mut transcript,
    )
    .expect("create_proof");
    transcript.finalize()
}

/// Verifies `proof` entirely on the CPU: the committed-instance commitments and the
/// final MSMs are evaluated with BLST directly, never through the GPU dispatch.
fn verifies_on_cpu(
    params: &ParamsKZG<Bls12>,
    vk: &VerifyingKey<Fq, Scheme>,
    circuits: &[FidelityCircuit],
    nb_committed: usize,
    proof: &[u8],
) -> bool {
    let blst_msm = |scalars: &[Fq], bases: &[G1Projective]| G1Projective::multi_exp(bases, scalars);

    let instances: Vec<[Vec<Fq>; 2]> = circuits.iter().map(|c| c.instances()).collect();
    let committed: Vec<Vec<G1Projective>> = instances
        .iter()
        .map(|cols| {
            cols[..nb_committed]
                .iter()
                .map(|col| blst_msm(col, &params.g_lagrange()[..col.len()]))
                .collect()
        })
        .collect();
    let committed: Vec<&[G1Projective]> = committed.iter().map(|c| c.as_slice()).collect();
    let plain: Vec<Vec<&[Fq]>> = instances
        .iter()
        .map(|cols| cols[nb_committed..].iter().map(|col| col.as_slice()).collect())
        .collect();
    let plain: Vec<&[&[Fq]]> = plain.iter().map(|cols| cols.as_slice()).collect();

    let mut transcript = CircuitTranscript::<Blake2bState>::init_from_bytes(proof);
    let Ok(guard) = prepare::<Fq, Scheme, _>(vk, &committed, &plain, &mut transcript) else {
        return false;
    };

    let msm = |terms: Vec<(&Fq, &G1Projective)>| {
        let (scalars, bases): (Vec<Fq>, Vec<G1Projective>) =
            terms.into_iter().map(|(s, b)| (*s, *b)).unzip();
        blst_msm(&scalars, &bases).to_affine()
    };
    let (left, right) = guard.split();
    let (left, right) = (msm(left), msm(right));
    let s_g2 = G2Prepared::from(params.s_g2().to_affine());
    let n_g2 = G2Prepared::from((-params.g2()).to_affine());
    Bls12::multi_miller_loop(&[(&left, &s_g2), (&right, &n_g2)])
        .final_exponentiation()
        .is_identity()
        .into()
}

fn fingerprint(bytes: &[u8]) -> String {
    let hash = blake2b_simd::Params::new().hash_length(16).hash(bytes);
    hash.to_hex().to_string()
}

/// `[len_le_u64 || bytes]` for each proof, so a mismatch can be attributed.
fn encode(proofs: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = vec![];
    for (_, proof) in proofs {
        out.extend((proof.len() as u64).to_le_bytes());
        out.extend(*proof);
    }
    out
}

fn decode(mut bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut proofs = vec![];
    while !bytes.is_empty() {
        let (len, rest) = bytes.split_at(8);
        let len = u64::from_le_bytes(len.try_into().unwrap()) as usize;
        let (proof, rest) = rest.split_at(len);
        proofs.push(proof.to_vec());
        bytes = rest;
    }
    proofs
}

#[test]
fn gpu_proof_fidelity() {
    let device = std::env::var("MIDNIGHT_DEVICE").unwrap_or_else(|_| "<unset>".into());
    eprintln!(
        "[fidelity] gpu feature: {}, MIDNIGHT_DEVICE={device}",
        cfg!(feature = "gpu")
    );
    #[cfg(feature = "gpu")]
    let gpu_forced = midnight_bls12_381_cuda::config::device_type()
        == midnight_bls12_381_cuda::config::DeviceType::Gpu;
    #[cfg(feature = "gpu")]
    assert!(
        !gpu_forced || midnight_bls12_381_cuda::is_gpu_available(),
        "MIDNIGHT_DEVICE=gpu but no usable GPU backend"
    );

    let params = ParamsKZG::<Bls12>::unsafe_setup(K, ChaCha8Rng::seed_from_u64(SETUP_SEED));
    let shape = FidelityCircuit::default();
    // `keygen_vk` would size k with the cost model, which does not support enabling
    // selectors while synthesizing later phases; k is fixed here anyway.
    let vk = keygen_vk_with_k::<_, Scheme, _>(&params, &shape, K).expect("keygen_vk");
    let pk = keygen_pk(vk, &shape).expect("keygen_pk");
    let vk = pk.get_vk();

    let single = [FidelityCircuit { x: 5, offset: 11 }];
    let multi = [
        FidelityCircuit { x: 7, offset: 3 },
        FidelityCircuit {
            x: 200,
            offset: 101,
        },
    ];

    let proof_single = prove(&params, &pk, &single, 1, PROVER_SEED);
    let proof_multi = prove(&params, &pk, &multi, 0, PROVER_SEED + 1);

    #[cfg(feature = "gpu")]
    {
        let fallbacks = midnight_bls12_381_cuda::config::gpu_fallback_count();
        eprintln!("[fidelity] GPU->CPU failure fallbacks while proving: {fallbacks}");
        if gpu_forced {
            assert_eq!(fallbacks, 0, "GPU run fell back to CPU");
        }
    }

    assert!(
        verifies_on_cpu(&params, vk, &single, 1, &proof_single),
        "proof single rejected"
    );
    assert!(
        verifies_on_cpu(&params, vk, &multi, 0, &proof_multi),
        "proof multi rejected"
    );
    // Negative control: the CPU check must reject a proof with one flipped bit
    // (in the last commitment, the KZG opening proof).
    let mut tampered = proof_single.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(
        !verifies_on_cpu(&params, vk, &single, 1, &tampered),
        "tampered proof accepted"
    );

    let proofs = [
        ("single", proof_single.as_slice()),
        ("multi", proof_multi.as_slice()),
    ];
    for (name, proof) in proofs {
        eprintln!(
            "[fidelity] proof {name}: {} bytes, blake2b-128 {}",
            proof.len(),
            fingerprint(proof)
        );
    }
    let encoded = encode(&proofs);

    if let Ok(path) = std::env::var("FIDELITY_DUMP") {
        std::fs::write(&path, &encoded).expect("write FIDELITY_DUMP");
        eprintln!("[fidelity] wrote {} bytes to {path}", encoded.len());
    }

    if let Ok(path) = std::env::var("FIDELITY_EXPECT") {
        let expected = std::fs::read(&path).expect("read FIDELITY_EXPECT");
        let expected = decode(&expected);
        assert_eq!(
            expected.len(),
            proofs.len(),
            "{path} holds a different number of proofs"
        );
        for ((name, proof), expected) in proofs.iter().zip(&expected) {
            if *proof != expected.as_slice() {
                let first_diff = proof
                    .iter()
                    .zip(expected)
                    .position(|(a, b)| a != b)
                    .unwrap_or(proof.len().min(expected.len()));
                panic!(
                    "proof {name} differs from {path}: {} vs {} bytes, first difference at \
                     byte {first_diff} (expected blake2b-128 {})",
                    proof.len(),
                    expected.len(),
                    fingerprint(expected)
                );
            }
        }
        eprintln!("[fidelity] proof bytes identical to {path}");
    }

    #[cfg(not(feature = "truncated-challenges"))]
    {
        for ((name, proof), (expected_name, expected)) in proofs.iter().zip(EXPECTED_FINGERPRINTS) {
            assert_eq!(*name, expected_name);
            assert_eq!(
                fingerprint(proof),
                expected,
                "proof {name} is not the upstream CPU prover's (blake2b-128 {expected}); if the \
                 upstream prover changed, update EXPECTED_FINGERPRINTS"
            );
        }
        eprintln!("[fidelity] proof fingerprints match the pinned upstream CPU bytes");
    }
}
