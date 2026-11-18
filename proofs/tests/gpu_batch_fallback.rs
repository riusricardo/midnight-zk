//! A batch transform the GPU cannot run falls back to the CPU and says so: the
//! fallback is counted and, under `MIDNIGHT_GPU_STRICT=1`, fatal; without
//! strict mode the result is the CPU's.
//!
//! The NTT domain is capped below the batch's polynomial size, so the GPU
//! batch fails at its domain initialization. The cap and the device mode are
//! read once per process, hence a binary of its own.
//!
//!   ICICLE_BACKEND_INSTALL_DIR=<install> MIDNIGHT_GPU_STRICT=1 \
//!     cargo test --release --features gpu --test gpu_batch_fallback
#![cfg(feature = "gpu")]

mod common;

use std::panic::{catch_unwind, AssertUnwindSafe};

use ff::{Field, PrimeField};
use midnight_bls12_381_cuda::config::{gpu_fallback_count, gpu_strict};
use midnight_curves::Fq;
use midnight_proofs::utils::fft::{best_fft_batch, ifft_batch};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const K: u32 = 13;
const POLYS: usize = 3;

#[test]
fn a_batch_the_gpu_cannot_run_is_reported_as_a_fallback() {
    std::env::set_var("MIDNIGHT_NTT_MAX_LOG", (K - 1).to_string());
    std::env::set_var("MIDNIGHT_DEVICE", "gpu");
    std::env::set_var("MIDNIGHT_AUTOTUNE", "0"); // forced GPU mode: no crossover to measure
    let Some(_gpu) = common::gpu_test() else { return };

    let n = 1usize << K;
    let mut omega = Fq::ROOT_OF_UNITY;
    for _ in K..Fq::S {
        omega = omega.square();
    }
    let omega_inv = omega.invert().unwrap();
    let divisor = Fq::from(n as u64).invert().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(K as u64);
    let input: Vec<Fq> = (0..POLYS * n).map(|_| Fq::random(&mut rng)).collect();

    type Batch = fn(&mut [Fq], usize, Fq, Fq);
    let forward: Batch = |b, n, omega, _| best_fft_batch(b, n, omega, K);
    let inverse: Batch = |b, n, omega_inv, divisor| ifft_batch(b, n, omega_inv, K, divisor);
    for (op, batch, root) in [("NTT (best_fft_batch)", forward, omega), ("iNTT (ifft_batch)", inverse, omega_inv)]
    {
        let mut data = input.clone();
        let fallbacks = gpu_fallback_count();
        let outcome = catch_unwind(AssertUnwindSafe(|| batch(&mut data, n, root, divisor)));
        assert_eq!(gpu_fallback_count(), fallbacks + 1, "{op}: the GPU failure was not reported");
        match outcome {
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or_default();
                assert!(gpu_strict() && message.contains(op), "{op}: unexpected panic: {message}");
            }
            Ok(()) => {
                assert!(!gpu_strict(), "{op}: strict mode let a fallback through");
                let mut want = input.clone();
                for poly in want.chunks_exact_mut(n) {
                    midnight_curves::fft::best_fft(poly, root, K);
                    if op.starts_with("iNTT") {
                        poly.iter_mut().for_each(|v| *v *= divisor);
                    }
                }
                assert!(data == want, "{op}: the CPU fallback's result is wrong");
            }
        }
    }
}
