/*
 * Copyright (C) 2026 Midnight Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! GPU-aware FFT operations
//!
//! This module provides FFT operations that automatically use GPU acceleration
//! when available for supported field types (specifically BLS12-381 scalar `Fq`).
//!
//! # Architecture
//!
//! The module uses the `dispatch_ntt_inplace` helper from `gpu_accel` to
//! automatically route to GPU or CPU based on:
//! - Field type (only `Fq` is GPU-accelerated)
//! - The routing rule (`MIDNIGHT_DEVICE`, the calibrated NTT crossover, and
//!   whether the GPU can be used)
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::utils::fft::{best_fft, ifft};
//!
//! // These automatically use GPU for Fq when beneficial
//! best_fft(&mut coeffs, omega, k);
//! ifft(&mut evals, omega_inv, k, divisor);
//! ```

use ff::PrimeField;

#[cfg(feature = "gpu")]
use crate::gpu_accel::dispatch_ntt_inplace;

/// Perform forward FFT (coefficients -> evaluations)
///
/// This is a drop-in replacement for `midnight_curves::fft::best_fft` that
/// automatically uses GPU when beneficial for `Fq`.
///
/// # Arguments
/// * `a` - Input/output slice (modified in-place)
/// * `omega` - Primitive root of unity for the domain
/// * `log_n` - Log2 of the domain size
#[inline]
pub fn best_fft<F: PrimeField + 'static>(a: &mut [F], omega: F, log_n: u32) {
    #[cfg(feature = "gpu")]
    {
        // Use the dispatch helper - it handles type checking and GPU/CPU routing
        let used_gpu = dispatch_ntt_inplace(
            a,
            |fq_slice| {
                crate::gpu_accel::backend::ntt::forward_ntt_inplace_auto(fq_slice)
                    .map_err(|e| e.to_string())
            },
            |data| midnight_curves::fft::best_fft(data, omega, log_n),
        );
        
        if used_gpu {
            #[cfg(feature = "trace-fft")]
            tracing::debug!("GPU forward FFT completed for size 2^{}", log_n);
            return;
        }
    }
    
    // CPU fallback (or non-GPU build)
    #[cfg(not(feature = "gpu"))]
    midnight_curves::fft::best_fft(a, omega, log_n);
}

/// Perform inverse FFT (evaluations -> coefficients)
///
/// This performs the FFT with omega_inv and then multiplies by the divisor (1/n).
///
/// # Arguments
/// * `a` - Input/output slice (modified in-place)
/// * `omega_inv` - Inverse of the primitive root of unity
/// * `log_n` - Log2 of the domain size
/// * `divisor` - Scaling factor (typically 1/n)
#[inline]
pub fn ifft<F: PrimeField + 'static>(a: &mut [F], omega_inv: F, log_n: u32, divisor: F) {
    #[cfg(feature = "gpu")]
    {
        // Use the dispatch helper for inverse NTT
        // Note: ICICLE's inverse NTT already includes the 1/n scaling
        let used_gpu = dispatch_ntt_inplace(
            a,
            |fq_slice| {
                crate::gpu_accel::backend::ntt::inverse_ntt_inplace_auto(fq_slice)
                    .map_err(|e| e.to_string())
            },
            |data| {
                // CPU path: do FFT then apply divisor
                midnight_curves::fft::best_fft(data, omega_inv, log_n);
                use super::arithmetic::parallelize;
                parallelize(data, |chunk, _| {
                    for val in chunk {
                        *val *= &divisor;
                    }
                });
            },
        );
        
        if used_gpu {
            #[cfg(feature = "trace-fft")]
            tracing::debug!("GPU inverse FFT completed for size 2^{}", log_n);
            return;
        }
    }
    
    // CPU fallback (or non-GPU build)
    #[cfg(not(feature = "gpu"))]
    {
        midnight_curves::fft::best_fft(a, omega_inv, log_n);
        
        // Apply divisor
        use super::arithmetic::parallelize;
        parallelize(a, |chunk, _| {
            for val in chunk {
                *val *= &divisor;
            }
        });
    }
}
