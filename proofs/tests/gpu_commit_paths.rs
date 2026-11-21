//! The KZG commitments the prover makes besides a single Lagrange commitment:
//! `commit_lagrange_batch` (every advice, lookup, permutation and trash batch)
//! with no polynomial, one, and several, and `commit` in coefficient form at
//! the domain's size and below it. Each is compared with BLST
//! (`common::blst_commit_*`), and its route is checked: in forced GPU mode
//! every MSM runs on the GPU. In Auto mode a batch gives the CPU a share
//! planned from costs the process measures, and a class of MSMs the CPU has
//! not run yet is probed with one MSM of a batch of at least four, so the first
//! batch a process commits puts exactly one MSM on the CPU. Hence a binary of
//! its own, with that test (the batches) and one that commits no batch.
//! Below the MSM crossover Auto mode commits on the CPU (a batch too) and
//! forced GPU mode on the GPU. Parameters downsized after a GPU commitment
//! commit with the smaller domain's bases.
//!
//!   ICICLE_BACKEND_INSTALL_DIR=<install> MIDNIGHT_GPU_STRICT=1 \
//!     cargo test --release --features gpu --test gpu_commit_paths
#![cfg(feature = "gpu")]

mod common;

use common::{blst_commit_coeffs, blst_commit_lagrange, gpu_msm_k, gpu_test};
use ff::Field;
use midnight_bls12_381_cuda::config::{device_type, gpu_fallback_count, DeviceType};
use midnight_curves::{Bls12, Fq, G1Projective};
use midnight_proofs::gpu_accel::backend::{
    self,
    diag::{gpu_op_count, OpKind},
};
use midnight_proofs::poly::{
    commitment::PolynomialCommitmentScheme,
    kzg::{params::ParamsKZG, KZGCommitmentScheme},
    EvaluationDomain, LagrangeCoeff, Polynomial,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const BATCH: usize = 5;

fn gpu_msms() -> u64 {
    gpu_op_count(backend::line_id(), OpKind::Msm)
}

#[test]
fn commitments_equal_blst_on_every_path() {
    let Some(_gpu) = gpu_test() else { return };
    // One above the MSM crossover, so half the domain still goes to the GPU in Auto mode.
    let k = gpu_msm_k(12) + 1;
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
    let domain = EvaluationDomain::<Fq>::new(1, k);
    let fallbacks = gpu_fallback_count();

    let polys: Vec<Polynomial<Fq, LagrangeCoeff>> = (0..BATCH)
        .map(|_| {
            let mut p = domain.empty_lagrange();
            p.iter_mut().for_each(|v| *v = Fq::random(&mut rng));
            p
        })
        .collect();
    let refs: Vec<&Polynomial<Fq, LagrangeCoeff>> = polys.iter().collect();
    let want: Vec<G1Projective> = polys.iter().map(|p| blst_commit_lagrange(&params, p)).collect();

    assert!(params.commit_lagrange_batch(&[]).is_empty(), "an empty batch commits nothing");

    let before = gpu_msms();
    assert_eq!(params.commit_lagrange_batch(&refs[..1]), want[..1], "a batch of one");
    assert_eq!(gpu_msms() - before, 1, "a batch of one is one GPU commitment");

    let before = gpu_msms();
    assert_eq!(params.commit_lagrange_batch(&refs), want, "a batch of {BATCH}");
    let on_gpu = gpu_msms() - before;
    if device_type() == DeviceType::Auto {
        assert_eq!(on_gpu, BATCH as u64 - 1, "Auto: the first batch probes the CPU with one MSM");
    } else {
        assert_eq!(on_gpu, BATCH as u64, "forced GPU: every MSM of the batch on the GPU");
    }

    // Coefficient form: 2^k coefficients, and 2^(k-1) (an MSM over a prefix of the bases).
    for log in [k, k - 1] {
        let coeffs: Vec<Fq> = (0..1usize << log).map(|_| Fq::random(&mut rng)).collect();
        let poly = EvaluationDomain::<Fq>::new(1, log).coeff_from_vec(coeffs.clone());
        let before = gpu_msms();
        assert_eq!(
            KZGCommitmentScheme::<Bls12>::commit(&params, &poly),
            blst_commit_coeffs(&params, &coeffs, k, domain.get_omega()),
            "commit of 2^{log} coefficients"
        );
        assert_eq!(gpu_msms() - before, 1, "commit of 2^{log} coefficients on the GPU");
    }

    // Below the MSM crossover: Auto mode commits on the CPU, and a batch takes
    // no batch path; forced GPU mode commits every MSM on the GPU.
    let log = midnight_bls12_381_cuda::min_gpu_size().trailing_zeros() - 1;
    let routed = |msms: u64| if device_type() == DeviceType::Gpu { msms } else { 0 };
    let small_domain = EvaluationDomain::<Fq>::new(1, log);
    let mut random = || (0..1usize << log).map(|_| Fq::random(&mut rng)).collect::<Vec<Fq>>();
    let coeffs = random();
    let before = gpu_msms();
    assert_eq!(
        KZGCommitmentScheme::<Bls12>::commit(&params, &small_domain.coeff_from_vec(coeffs.clone())),
        blst_commit_coeffs(&params, &coeffs, k, domain.get_omega()),
        "commit of 2^{log} coefficients"
    );
    assert_eq!(gpu_msms() - before, routed(1), "commit of 2^{log} coefficients");
    let small: Vec<Polynomial<Fq, LagrangeCoeff>> =
        (0..BATCH).map(|_| small_domain.lagrange_from_vec(random())).collect();
    let want: Vec<G1Projective> = small.iter().map(|p| blst_commit_lagrange(&params, p)).collect();
    let before = gpu_msms();
    assert_eq!(KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &small[0]), want[0]);
    assert_eq!(gpu_msms() - before, routed(1), "commit_lagrange of 2^{log} values");
    let refs: Vec<&Polynomial<Fq, LagrangeCoeff>> = small.iter().collect();
    let before = gpu_msms();
    assert_eq!(params.commit_lagrange_batch(&refs), want, "a batch of 2^{log} values each");
    assert_eq!(gpu_msms() - before, routed(BATCH as u64), "a batch of 2^{log} values each");
    assert_eq!(gpu_fallback_count(), fallbacks, "GPU work fell back to the CPU");
}

/// Downsizing parameters re-derives the Lagrange bases for the smaller domain
/// and drops the device copies uploaded for the larger one (fix C2): a GPU
/// commitment after it uses the bases of parameters set up at the smaller size.
#[test]
fn a_commitment_after_downsizing_uses_the_smaller_domain() {
    let Some(_gpu) = gpu_test() else { return };
    let k = gpu_msm_k(12) + 1;
    let mut params = ParamsKZG::<Bls12>::unsafe_setup(k, ChaCha8Rng::seed_from_u64(12));
    let smaller = ParamsKZG::<Bls12>::unsafe_setup(k - 1, ChaCha8Rng::seed_from_u64(12));
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let mut random = |log: u32| {
        EvaluationDomain::<Fq>::new(1, log).lagrange_from_vec((0..1usize << log).map(|_| Fq::random(&mut rng)).collect())
    };
    // The larger domain's bases go to the device with this commitment.
    let large = random(k);
    assert_eq!(KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &large), blst_commit_lagrange(&params, &large));
    params.downsize(k - 1);
    assert!(params.g_lagrange() == smaller.g_lagrange(), "the downsized Lagrange bases");
    let poly = random(k - 1);
    let before = gpu_msms();
    assert_eq!(
        KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &poly),
        blst_commit_lagrange(&smaller, &poly),
        "a commitment after downsizing"
    );
    assert_eq!(gpu_msms() - before, 1, "a commitment after downsizing on the GPU");
}
