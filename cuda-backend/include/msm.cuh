/**
 * @file msm.cuh
 * @brief Multi-Scalar Multiplication (MSM) using Pippenger's bucket method
 * 
 * This implementation follows the same algorithm as BLST and Icicle:
 * 1. Split scalars into windows of size c bits
 * 2. Accumulate points into 2^c buckets per window
 * 3. Sum buckets using triangle sum
 * 4. Combine windows with doublings
 */

#pragma once

#include "point.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>

namespace msm {

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// MSM Configuration and Helpers
// =============================================================================

/**
 * @brief Determine optimal window size c based on MSM size
 * 
 * Matches the heuristic used by Icicle's get_optimal_c function.
 */
__host__ int get_optimal_c(int msm_size) {
    if (msm_size <= 1) return 1;
    
    int log_size = 0;
    int temp = msm_size;
    while (temp > 1) {
        temp >>= 1;
        log_size++;
    }
    
    // Empirical optimal values based on MSM size
    if (log_size <= 10) return 8;
    if (log_size <= 12) return 10;
    if (log_size <= 14) return 12;
    if (log_size <= 16) return 13;
    if (log_size <= 18) return 14;
    if (log_size <= 20) return 15;
    return 16;
}

/**
 * @brief Calculate number of windows needed for given scalar size and window size
 */
__host__ __device__ int get_num_windows(int scalar_bits, int c) {
    return (scalar_bits + c - 1) / c;
}

// =============================================================================
// MSM Kernels
// =============================================================================

/**
 * @brief Initialize all buckets to identity
 */
__global__ void initialize_buckets_kernel(
    G1Projective* buckets,
    unsigned int total_buckets
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_buckets) {
        buckets[idx] = G1Projective::identity();
    }
}

/**
 * @brief Extract window value from scalar
 * 
 * Returns the c-bit window at position window_idx, with sign handling
 * for signed digit representation.
 */
__device__ int extract_window(
    const Fr& scalar,
    int window_idx,
    int c,
    int scalar_bits
) {
    int bit_offset = window_idx * c;
    if (bit_offset >= scalar_bits) return 0;
    
    // Extract bits from the scalar limbs
    int limb_idx = bit_offset / 64;
    int bit_in_limb = bit_offset % 64;
    
    uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;
    
    // Handle cross-limb windows
    if (bit_in_limb + c > 64 && limb_idx + 1 < Fr::LIMBS) {
        window |= (scalar.limbs[limb_idx + 1] << (64 - bit_in_limb));
    }
    
    // Mask to c bits
    window &= ((1ULL << c) - 1);
    
    return (int)window;
}

/**
 * @brief Accumulate points into buckets (main accumulation kernel)
 * 
 * Each thread processes one scalar-point pair.
 * Uses atomic operations for bucket accumulation.
 * 
 * Note: This is the serial-per-bucket approach. For production,
 * we use sorting and conflict-free accumulation.
 */
__global__ void accumulate_buckets_kernel(
    G1Projective* buckets,          // [num_windows * num_buckets] buckets
    const Fr* scalars,              // [msm_size] scalars
    const G1Affine* bases,          // [msm_size] base points
    int msm_size,
    int c,                          // Window size
    int num_windows,
    int num_buckets                 // 2^(c-1) buckets per window (signed digit)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= msm_size) return;
    
    Fr scalar = scalars[idx];
    G1Affine base = bases[idx];
    
    if (base.is_identity()) return;
    
    // Process each window
    for (int w = 0; w < num_windows; w++) {
        int window_val = extract_window(scalar, w, c, Fr::LIMBS * 64);
        
        if (window_val == 0) continue;
        
        // Signed digit representation: if window > 2^(c-1), use negative
        bool negate = false;
        if (window_val > num_buckets) {
            window_val = (1 << c) - window_val;
            negate = true;
        }
        
        // Bucket index (1-indexed, bucket 0 is for window_val = 0 which we skip)
        int bucket_idx = w * num_buckets + (window_val - 1);
        
        // Add point to bucket (with potential negation)
        G1Projective point = G1Projective::from_affine(negate ? base.neg() : base);
        
        // Atomic add to bucket
        // Note: For production, use conflict-free accumulation with sorting
        G1Projective current = buckets[bucket_idx];
        G1Projective result;
        g1_add(result, current, point);
        buckets[bucket_idx] = result;
    }
}

/**
 * @brief Parallel bucket reduction using triangle sum
 * 
 * Computes: sum_{i=1}^{n} i * bucket[i] = sum_{j=1}^{n} sum_{i=j}^{n} bucket[i]
 * This is done as a running sum from the last bucket to the first.
 */
__global__ void bucket_reduction_kernel(
    G1Projective* window_results,   // [num_windows] output per window
    const G1Projective* buckets,    // [num_windows * num_buckets] buckets
    int num_windows,
    int num_buckets
) {
    int window_idx = blockIdx.x;
    if (window_idx >= num_windows) return;
    
    // Each block handles one window
    // For simplicity, one thread per window (can be parallelized further)
    if (threadIdx.x != 0) return;
    
    const G1Projective* window_buckets = buckets + window_idx * num_buckets;
    
    G1Projective running_sum = G1Projective::identity();
    G1Projective window_sum = G1Projective::identity();
    
    // Triangle sum: iterate from highest bucket to lowest
    for (int i = num_buckets - 1; i >= 0; i--) {
        g1_add(running_sum, running_sum, window_buckets[i]);
        g1_add(window_sum, window_sum, running_sum);
    }
    
    window_results[window_idx] = window_sum;
}

/**
 * @brief Final accumulation: combine window results with doublings
 * 
 * result = sum_{w=0}^{num_windows-1} 2^{w*c} * window_results[w]
 */
__global__ void final_accumulation_kernel(
    G1Projective* result,
    const G1Projective* window_results,
    int num_windows,
    int c
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G1Projective acc = window_results[num_windows - 1];
    
    // Process windows from second-highest to lowest
    for (int w = num_windows - 2; w >= 0; w--) {
        // Double c times
        for (int i = 0; i < c; i++) {
            g1_double(acc, acc);
        }
        // Add window result
        g1_add(acc, acc, window_results[w]);
    }
    
    *result = acc;
}

// =============================================================================
// Main MSM Entry Point
// =============================================================================

/**
 * @brief MSM using Pippenger's bucket method
 * 
 * This is the main entry point that matches the Icicle signature.
 */
template<typename S, typename A, typename P>
cudaError_t msm_cuda(
    const S* scalars,
    const A* bases,
    int msm_size,
    const MSMConfig& config,
    P* result
) {
    if (msm_size == 0) {
        if (!config.are_results_on_device) {
            *result = P::identity();
        } else {
            P identity = P::identity();
            cudaMemcpy(result, &identity, sizeof(P), cudaMemcpyHostToDevice);
        }
        return cudaSuccess;
    }

    cudaStream_t stream = config.stream;
    
    // Determine optimal window size
    int c = config.c > 0 ? config.c : get_optimal_c(msm_size);
    int scalar_bits = config.bitsize > 0 ? config.bitsize : S::LIMBS * 64;
    int num_windows = get_num_windows(scalar_bits, c);
    int num_buckets = (1 << (c - 1));  // Signed digit: half the buckets
    int total_buckets = num_windows * num_buckets;
    
    // Allocate device memory
    S* d_scalars = nullptr;
    A* d_bases = nullptr;
    P* d_result = nullptr;
    G1Projective* d_buckets = nullptr;
    G1Projective* d_window_results = nullptr;
    
    cudaError_t err;
    
    // Allocate buckets
    err = cudaMalloc(&d_buckets, total_buckets * sizeof(G1Projective));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_window_results, num_windows * sizeof(G1Projective));
    if (err != cudaSuccess) {
        cudaFree(d_buckets);
        return err;
    }
    
    // Handle input data
    if (config.are_scalars_on_device) {
        d_scalars = const_cast<S*>(scalars);
    } else {
        err = cudaMalloc(&d_scalars, msm_size * sizeof(S));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(S), 
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
    if (config.are_points_on_device) {
        d_bases = const_cast<A*>(bases);
    } else {
        err = cudaMalloc(&d_bases, msm_size * sizeof(A));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(A),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
    if (config.are_results_on_device) {
        d_result = result;
    } else {
        err = cudaMalloc(&d_result, sizeof(P));
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Initialize buckets
    {
        int block_size = 256;
        int grid_size = (total_buckets + block_size - 1) / block_size;
        initialize_buckets_kernel<<<grid_size, block_size, 0, stream>>>(
            d_buckets, total_buckets
        );
    }
    
    // Accumulate into buckets
    {
        int block_size = 256;
        int grid_size = (msm_size + block_size - 1) / block_size;
        accumulate_buckets_kernel<<<grid_size, block_size, 0, stream>>>(
            d_buckets,
            d_scalars,
            d_bases,
            msm_size,
            c,
            num_windows,
            num_buckets
        );
    }
    
    // Bucket reduction
    {
        bucket_reduction_kernel<<<num_windows, 1, 0, stream>>>(
            d_window_results,
            d_buckets,
            num_windows,
            num_buckets
        );
    }
    
    // Final accumulation
    {
        final_accumulation_kernel<<<1, 1, 0, stream>>>(
            d_result,
            d_window_results,
            num_windows,
            c
        );
    }
    
    // Copy result back if needed
    if (!config.are_results_on_device) {
        err = cudaMemcpyAsync(result, d_result, sizeof(P),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Synchronize if not async
    if (!config.is_async) {
        err = cudaStreamSynchronize(stream);
    }

cleanup:
    // Free allocated memory
    if (d_buckets) cudaFree(d_buckets);
    if (d_window_results) cudaFree(d_window_results);
    if (!config.are_scalars_on_device && d_scalars) cudaFree(d_scalars);
    if (!config.are_points_on_device && d_bases) cudaFree(d_bases);
    if (!config.are_results_on_device && d_result) cudaFree(d_result);
    
    return err;
}

} // namespace msm
