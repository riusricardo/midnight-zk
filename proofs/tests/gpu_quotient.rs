//! The quotient numerator on the GPU (`plonk/gpu_eval.rs`) over the circuit
//! shapes its column layout and program compiler handle: no copy constraints,
//! one equality column, one permutation set, several sets with a partial last
//! one, no, one and two lookups and trashcans, and two circuits in one proof.
//! Each proof must run the GPU evaluation (the backend's GraphEval counter, no
//! fallback), leave the key's columns cached on the device, and be accepted by
//! the verifier's pairing computed with BLST alone: a wrong numerator gives a
//! quotient the verifier rejects.
//!
//!   ICICLE_BACKEND_INSTALL_DIR=<install> MIDNIGHT_GPU_STRICT=1 \
//!     cargo test --release --features gpu --test gpu_quotient
#![cfg(feature = "gpu")]

mod common;

use blake2b_simd::State as Blake2bState;
use common::{expect_gpu_ops, gpu_ntt_k, gpu_test, Verdict};
use midnight_curves::{Bls12, Fq, G1Projective};
use midnight_proofs::{
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    gpu_accel::backend::diag::OpKind,
    plonk::{
        create_proof, keygen_pk, keygen_vk_with_k, prepare, Advice, Circuit, Column, ConstraintSystem,
        Constraints, Error, Expression, Selector, TableColumn,
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

/// Witnesses are `(t * 37 + offset) mod PERIOD` of `t = row + column`: a cell
/// equals the one PERIOD rows below it and, in the next column, the one a row
/// above it (copy constraints between equal values), and every value lies in
/// the lookup table `0..PERIOD`.
const PERIOD: usize = 64;

#[derive(Clone, Debug)]
struct ShapeConfig {
    x: Vec<Column<Advice>>,
    s: Column<Advice>,
    p: Vec<Column<Advice>>,
    table: TableColumn,
    q: Selector,
    q_public: Selector,
    q_lookup: Selector,
    q_trash: Vec<Selector>,
}

/// `EQ` advice columns with copy constraints (among at least two advice columns
/// `x`), `LOOKUPS` range lookups of `x_0, x_1, ..`, `TRASH` trashcans (gates
/// with additive selectors `p_t = x_0 x_1 + t + 1`), a plain gate
/// `s = x_0 + x_1` and the public input `x_0` of row 0.
#[derive(Clone, Debug)]
struct ShapeCircuit<const EQ: usize, const LOOKUPS: usize, const TRASH: usize> {
    rows: usize,
    offset: u64,
}

impl<const EQ: usize, const LOOKUPS: usize, const TRASH: usize> ShapeCircuit<EQ, LOOKUPS, TRASH> {
    fn x(&self, column: usize, row: usize) -> u64 {
        ((row + column) as u64 * 37 + self.offset) % PERIOD as u64
    }
}

impl<const EQ: usize, const LOOKUPS: usize, const TRASH: usize> Circuit<Fq> for ShapeCircuit<EQ, LOOKUPS, TRASH> {
    type Config = ShapeConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fq>) -> ShapeConfig {
        let public = meta.instance_column();
        let x: Vec<Column<Advice>> = (0..EQ.max(2)).map(|_| meta.advice_column()).collect();
        let s = meta.advice_column();
        let p: Vec<Column<Advice>> = (0..TRASH).map(|_| meta.advice_column()).collect();
        for &column in &x[..EQ] {
            meta.enable_equality(column);
        }
        let table = meta.lookup_table_column();
        let q = meta.selector();
        let q_public = meta.selector();
        let q_lookup = meta.complex_selector();
        // Additive selectors (trashcans) must be complex selectors.
        let q_trash: Vec<Selector> = (0..TRASH).map(|_| meta.complex_selector()).collect();

        let (x0, x1) = (x[0], x[1]);
        meta.create_gate("s = x0 + x1", |meta| {
            let (a, b) = (meta.query_advice(x0, Rotation::cur()), meta.query_advice(x1, Rotation::cur()));
            let s = meta.query_advice(s, Rotation::cur());
            Constraints::with_selector(q, vec![a + b - s])
        });
        meta.create_gate("public = x0", |meta| {
            let public = meta.query_instance(public, Rotation::cur());
            let a = meta.query_advice(x0, Rotation::cur());
            Constraints::with_selector(q_public, vec![public - a])
        });
        for (t, (&p_t, &q_t)) in p.iter().zip(&q_trash).enumerate() {
            meta.create_gate("p = x0 * x1 + t + 1", |meta| {
                let (a, b) = (meta.query_advice(x0, Rotation::cur()), meta.query_advice(x1, Rotation::cur()));
                let p = meta.query_advice(p_t, Rotation::cur());
                let t = Expression::Constant(Fq::from(t as u64 + 1));
                Constraints::with_additive_selector(q_t, vec![a * b + t - p])
            });
        }
        for &column in &x[..LOOKUPS] {
            meta.lookup("range", |meta| {
                let q = meta.query_selector(q_lookup);
                let v = meta.query_advice(column, Rotation::cur());
                vec![(q * v, table)]
            });
        }
        ShapeConfig { x, s, p, table, q, q_public, q_lookup, q_trash }
    }

    fn synthesize(&self, config: ShapeConfig, mut layouter: impl Layouter<Fq>) -> Result<(), Error> {
        layouter.assign_table(
            || "range table",
            |mut table| {
                for v in 0..PERIOD {
                    table.assign_cell(|| "range", config.table, v, || Value::known(Fq::from(v as u64)))?;
                }
                Ok(())
            },
        )?;
        layouter.assign_region(
            || "main",
            |mut region| {
                let mut cells: Vec<Vec<AssignedCell<Fq, Fq>>> = vec![Vec::new(); EQ];
                config.q_public.enable(&mut region, 0)?;
                for i in 0..self.rows {
                    config.q.enable(&mut region, i)?;
                    if LOOKUPS > 0 {
                        config.q_lookup.enable(&mut region, i)?;
                    }
                    for q_t in &config.q_trash {
                        q_t.enable(&mut region, i)?;
                    }
                    let x: Vec<u64> = (0..config.x.len()).map(|j| self.x(j, i)).collect();
                    for (j, &column) in config.x.iter().enumerate() {
                        let cell = region.assign_advice(|| "x", column, i, || Value::known(Fq::from(x[j])))?;
                        if j < EQ {
                            cells[j].push(cell);
                        }
                    }
                    region.assign_advice(|| "s", config.s, i, || Value::known(Fq::from(x[0] + x[1])))?;
                    for (t, &column) in config.p.iter().enumerate() {
                        let p = Fq::from(x[0] * x[1] + t as u64 + 1);
                        region.assign_advice(|| "p", column, i, || Value::known(p))?;
                    }
                }
                for j in 0..EQ {
                    for i in 0..self.rows - PERIOD {
                        region.constrain_equal(cells[j][i].cell(), cells[j][i + PERIOD].cell())?;
                    }
                    if j + 1 < EQ {
                        for i in 0..self.rows - 1 {
                            region.constrain_equal(cells[j][i + 1].cell(), cells[j + 1][i].cell())?;
                        }
                    }
                }
                Ok(())
            },
        )
    }
}

/// The permutation argument's sets: their number, and whether the last one is
/// partial (fewer columns than the others).
fn permutation_sets(cs: &ConstraintSystem<Fq>) -> (usize, bool) {
    let (columns, chunk) = (cs.permutation().get_columns().len(), cs.degree() - 2);
    (columns.div_ceil(chunk), columns % chunk != 0)
}

/// Proves `circuits` circuits of the shape on the GPU and checks the proof
/// with BLST; returns the shape's permutation sets for the caller's claim.
fn prove_shape<const EQ: usize, const LOOKUPS: usize, const TRASH: usize>(circuits: usize) -> Option<(usize, bool)> {
    let _gpu = gpu_test()?;
    let k = gpu_ntt_k(10);
    let rows = 1usize << (k - 1);
    let params = ParamsKZG::<Bls12>::unsafe_setup(k, ChaCha8Rng::seed_from_u64(1));
    let shape = ShapeCircuit::<EQ, LOOKUPS, TRASH> { rows, offset: 0 };
    let vk = keygen_vk_with_k::<_, Scheme, _>(&params, &shape, k).expect("keygen_vk");
    let pk = keygen_pk(vk, &shape).expect("keygen_pk");
    let vk = pk.get_vk();
    let cs = vk.cs();
    assert_eq!(
        (cs.permutation().get_columns().len(), cs.lookups().len(), cs.trashcans().len()),
        (EQ, LOOKUPS, TRASH),
        "the shape's arguments"
    );

    let circuits: Vec<ShapeCircuit<EQ, LOOKUPS, TRASH>> =
        (0..circuits).map(|c| ShapeCircuit { rows, offset: 5 * c as u64 + 3 }).collect();
    let publics: Vec<Vec<Fq>> = circuits.iter().map(|c| vec![Fq::from(c.x(0, 0))]).collect();
    let instances: Vec<Vec<&[Fq]>> = publics.iter().map(|p| vec![p.as_slice()]).collect();
    let instances: Vec<&[&[Fq]]> = instances.iter().map(|i| i.as_slice()).collect();
    let proof = expect_gpu_ops(&[OpKind::GraphEval], || {
        let mut transcript = CircuitTranscript::<Blake2bState>::init();
        create_proof::<Fq, Scheme, _, _>(
            &params,
            &pk,
            &circuits,
            0,
            &instances,
            ChaCha8Rng::seed_from_u64(2),
            &mut transcript,
        )
        .expect("create_proof");
        transcript.finalize()
    });
    assert!(format!("{pk:?}").contains("KeyColumns { cached: true }"), "the key's columns are not cached");

    let committed: Vec<&[G1Projective]> = vec![&[]; circuits.len()];
    let mut transcript = CircuitTranscript::<Blake2bState>::init_from_bytes(&proof);
    let verdict = Verdict::of(&params, prepare::<Fq, Scheme, _>(vk, &committed, &instances, &mut transcript));
    assert_eq!(verdict, Verdict::Accepted, "the proof");
    Some(permutation_sets(cs))
}

#[test]
fn gates_only() {
    if let Some(sets) = prove_shape::<0, 0, 0>(1) {
        assert_eq!(sets, (0, false));
    }
}

#[test]
fn one_equality_column_and_a_lookup() {
    if let Some(sets) = prove_shape::<1, 1, 0>(1) {
        assert_eq!(sets.0, 1);
    }
}

#[test]
fn one_permutation_set_two_lookups_two_trashcans() {
    if let Some(sets) = prove_shape::<2, 2, 2>(1) {
        assert_eq!(sets.0, 1);
    }
}

#[test]
fn several_permutation_sets_the_last_partial() {
    if let Some((sets, partial)) = prove_shape::<7, 1, 1>(1) {
        assert!(sets >= 2 && partial, "{sets} sets, last partial: {partial}");
    }
}

#[test]
fn two_circuits_in_one_proof() {
    prove_shape::<3, 1, 1>(2);
}
