//! The prover's commitments over precomputed bases (`MIDNIGHT_MSM_PRECOMPUTE`
//! = 2, opt-in): the bases are uploaded with a shifted copy, and an MSM over
//! them must cover every base, so `commit` pads a polynomial shorter than the
//! domain with zero coefficients (the multi-open quotient has n - 1, on every
//! proof). Coefficient form at the domain's size and below it, one Lagrange
//! commitment, and a batch, each compared with BLST (`common::blst_commit_*`)
//! and run on the GPU. The factor is read once per process, hence a binary of
//! its own that sets it first.
//!
//!   ICICLE_BACKEND_INSTALL_DIR=<install> MIDNIGHT_GPU_STRICT=1 \
//!     cargo test --release --features gpu --test gpu_commit_precompute
#![cfg(feature = "gpu")]

mod common;

use common::{blst_commit_coeffs, blst_commit_lagrange, expect_gpu_ops, gpu_msm_k, gpu_test};
use ff::Field;
use midnight_bls12_381_cuda::config::msm_precompute_factor;
use midnight_curves::{Bls12, Fq, G1Projective};
use midnight_proofs::gpu_accel::backend::diag::OpKind;
use midnight_proofs::poly::{
    commitment::PolynomialCommitmentScheme,
    kzg::{params::ParamsKZG, KZGCommitmentScheme},
    EvaluationDomain, LagrangeCoeff, Polynomial,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn precomputed_bases_commit_as_blst() {
    std::env::set_var("MIDNIGHT_MSM_PRECOMPUTE", "2");
    let Some(_gpu) = gpu_test() else { return };
    assert_eq!(msm_precompute_factor(), 2, "the precompute factor must be in effect");
    let k = gpu_msm_k(12) + 1;
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let params = ParamsKZG::<Bls12>::unsafe_setup(k, &mut rng);
    let domain = EvaluationDomain::<Fq>::new(1, k);

    for log in [k, k - 1] {
        let coeffs: Vec<Fq> = (0..1usize << log).map(|_| Fq::random(&mut rng)).collect();
        let poly = EvaluationDomain::<Fq>::new(1, log).coeff_from_vec(coeffs.clone());
        let got = expect_gpu_ops(&[OpKind::Msm], || KZGCommitmentScheme::<Bls12>::commit(&params, &poly));
        assert_eq!(
            got,
            blst_commit_coeffs(&params, &coeffs, k, domain.get_omega()),
            "commit of 2^{log} coefficients over precomputed bases"
        );
    }

    let polys: Vec<Polynomial<Fq, LagrangeCoeff>> = (0..5)
        .map(|_| {
            let mut p = domain.empty_lagrange();
            p.iter_mut().for_each(|v| *v = Fq::random(&mut rng));
            p
        })
        .collect();
    let want: Vec<G1Projective> = polys.iter().map(|p| blst_commit_lagrange(&params, p)).collect();
    let one = expect_gpu_ops(&[OpKind::Msm], || KZGCommitmentScheme::<Bls12>::commit_lagrange(&params, &polys[0]));
    assert_eq!(one, want[0], "commit_lagrange over precomputed bases");
    let refs: Vec<&Polynomial<Fq, LagrangeCoeff>> = polys.iter().collect();
    let batch = expect_gpu_ops(&[OpKind::Msm], || params.commit_lagrange_batch(&refs));
    assert_eq!(batch, want, "commit_lagrange_batch over precomputed bases");
}
