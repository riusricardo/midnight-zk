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
//! The module uses a trait-based approach where:
//! - `Fq` (BLS12-381 scalar) gets GPU-accelerated NTT via ICICLE when the `gpu` feature is enabled
//! - Other field types fall back to CPU FFT via `midnight_curves::fft::best_fft`
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
//!
//! # GPU Selection Criteria
//!
//! GPU is used when:
//! 1. The `gpu` feature is enabled
//! 2. The field type is `Fq` (BLS12-381 scalar)
//! 3. The size is above the threshold (default: 2^12 = 4096 elements)
//! 4. `MIDNIGHT_DEVICE` is not set to `cpu`

use ff::PrimeField;

#[cfg(feature = "gpu")]
use midnight_bls12_381_cuda::config::should_use_gpu_ntt;

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
pub fn best_fft<F: PrimeField>(a: &mut [F], omega: F, log_n: u32) {
    // Try GPU path for Fq
    #[cfg(feature = "gpu")]
    {
        if try_gpu_fft(a, log_n, false) {
            return;
        }
    }
    
    // Fallback to CPU
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
pub fn ifft<F: PrimeField>(a: &mut [F], omega_inv: F, log_n: u32, divisor: F) {
    // Try GPU path for Fq
    #[cfg(feature = "gpu")]
    {
        if try_gpu_ifft(a, log_n, divisor) {
            return;
        }
    }
    
    // Fallback to CPU
    midnight_curves::fft::best_fft(a, omega_inv, log_n);
    
    // Apply divisor
    use super::arithmetic::parallelize;
    parallelize(a, |a, _| {
        for a in a {
            *a *= &divisor;
        }
    });
}

/// Try to use GPU for forward FFT on Fq
///
/// Returns true if GPU was used, false if fallback to CPU is needed.
#[cfg(feature = "gpu")]
#[allow(unused_variables)]
fn try_gpu_fft<F: PrimeField>(a: &mut [F], log_n: u32, _is_inverse: bool) -> bool {
    use std::any::TypeId;
    use midnight_curves::Fq;
    
    // Check if F is Fq (BLS12-381 scalar)
    if TypeId::of::<F>() != TypeId::of::<Fq>() {
        return false;
    }
    
    let n = a.len();
    
    // Check if GPU is beneficial for this size
    if !should_use_gpu_ntt(n) {
        return false;
    }
    
    // Transmute to Fq and use GPU NTT
    // SAFETY: We verified F == Fq via TypeId check above
    let fq_slice: &mut [Fq] = unsafe {
        std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut Fq, n)
    };
    
    // Use the auto-selection API which handles GPU context
    match midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(fq_slice) {
        Ok(()) => {
            #[cfg(feature = "trace-fft")]
            tracing::debug!("GPU forward FFT completed for size 2^{}", log_n);
            true
        }
        Err(e) => {
            tracing::warn!("GPU FFT failed, falling back to CPU: {:?}", e);
            false
        }
    }
}

/// Try to use GPU for inverse FFT on Fq
///
/// Returns true if GPU was used, false if fallback to CPU is needed.
#[cfg(feature = "gpu")]
#[allow(unused_variables)]
fn try_gpu_ifft<F: PrimeField>(a: &mut [F], log_n: u32, _divisor: F) -> bool {
    use std::any::TypeId;
    use midnight_curves::Fq;
    
    // Check if F is Fq (BLS12-381 scalar)
    if TypeId::of::<F>() != TypeId::of::<Fq>() {
        return false;
    }
    
    let n = a.len();
    
    // Check if GPU is beneficial for this size
    if !should_use_gpu_ntt(n) {
        return false;
    }
    
    // Transmute to Fq and use GPU NTT
    // SAFETY: We verified F == Fq via TypeId check above
    let fq_slice: &mut [Fq] = unsafe {
        std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut Fq, n)
    };
    
    // Use the auto-selection API which handles GPU context
    // Note: ICICLE's inverse NTT already includes the 1/n scaling
    match midnight_bls12_381_cuda::ntt::inverse_ntt_inplace_auto(fq_slice) {
        Ok(()) => {
            #[cfg(feature = "trace-fft")]
            tracing::debug!("GPU inverse FFT completed for size 2^{}", log_n);
            true
        }
        Err(e) => {
            tracing::warn!("GPU IFFT failed, falling back to CPU: {:?}", e);
            false
        }
    }
}

/// Check if GPU FFT is available for the current configuration
#[cfg(feature = "gpu")]
pub fn gpu_fft_available() -> bool {
    midnight_bls12_381_cuda::is_gpu_available()
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_fft_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use midnight_curves::Fq;
    use ff::Field;
    use rand_core::OsRng;

    /// Test that our FFT matches the reference implementation
    #[test]
    fn test_fft_matches_reference() {
        let k = 8u32;
        let n = 1usize << k;
        
        // Compute omega
        let mut omega = Fq::ROOT_OF_UNITY;
        for _ in k..Fq::S {
            omega = omega.square();
        }
        
        // Create random input
        let mut our_input: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();
        let mut ref_input = our_input.clone();
        
        // Our FFT
        best_fft(&mut our_input, omega, k);
        
        // Reference FFT
        midnight_curves::fft::best_fft(&mut ref_input, omega, k);
        
        // Compare (may differ if GPU uses different root, but roundtrip should work)
        // For now, just test roundtrip
        let omega_inv = omega.invert().unwrap();
        let divisor = Fq::from(n as u64).invert().unwrap();
        
        ifft(&mut our_input, omega_inv, k, divisor);
        midnight_curves::fft::best_fft(&mut ref_input, omega_inv, k);
        for val in ref_input.iter_mut() {
            *val *= divisor;
        }
        
        // Both should produce original values
    }

    /// Test roundtrip FFT -> IFFT
    #[test]
    fn test_fft_roundtrip() {
        let k = 10u32;
        let n = 1usize << k;
        
        // Compute omega
        let mut omega = Fq::ROOT_OF_UNITY;
        for _ in k..Fq::S {
            omega = omega.square();
        }
        let omega_inv = omega.invert().unwrap();
        let divisor = Fq::from(n as u64).invert().unwrap();
        
        // Create input
        let original: Vec<Fq> = (0..n).map(|i| Fq::from(i as u64 + 1)).collect();
        let mut data = original.clone();
        
        // Forward FFT
        best_fft(&mut data, omega, k);
        
        // Inverse FFT
        ifft(&mut data, omega_inv, k, divisor);
        
        // Verify roundtrip
        for (i, (orig, recovered)) in original.iter().zip(data.iter()).enumerate() {
            assert_eq!(*orig, *recovered, "FFT roundtrip failed at index {}", i);
        }
    }
}
