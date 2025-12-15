/**
 * @file msm.cuh
 * @brief Multi-Scalar Multiplication (MSM) using Pippenger's bucket method
 * 
 * Production-ready implementation with:
 * - Conflict-free bucket accumulation (each bucket processed by single thread)
 * - Parallel point accumulation within buckets
 * - Optimal window size selection
 * - Signed digit representation for bucket reduction
 * 
 * Algorithm:
 * 1. Split scalars into windows of size c bits
 * 2. Each bucket is handled by one block - threads scan all points
 * 3. Triangle sum for weighted bucket summation
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
// MSM Configuration
// =============================================================================

/**
 * @brief Determine optimal window size based on MSM size
 * 
 * Empirically tuned values matching Icicle/BLST heuristics.
 * Larger windows = fewer additions but more buckets.
 */
__host__ __device__ inline int get_optimal_c(int msm_size) {
    if (msm_size <= 1) return 1;
    
    int log_size = 0;
    int temp = msm_size;
    while (temp > 1) {
        temp >>= 1;
        log_size++;
    }
    
    // Empirical optimal values
    if (log_size <= 8)  return 7;
    if (log_size <= 10) return 8;
    if (log_size <= 12) return 10;
    if (log_size <= 14) return 12;
    if (log_size <= 16) return 13;
    if (log_size <= 18) return 14;
    if (log_size <= 20) return 15;
    return 16;
}

/**
 * @brief Calculate number of windows for given parameters
 */
__host__ __device__ inline int get_num_windows(int scalar_bits, int c) {
    return (scalar_bits + c - 1) / c;
}

// =============================================================================
// Scalar Window Extraction
// =============================================================================

/**
 * @brief Extract c-bit window from scalar
 */
__device__ __forceinline__ int extract_window_value(
    const Fr& scalar,
    int window_idx,
    int c
) {
    int bit_offset = window_idx * c;
    int limb_idx = bit_offset / 64;
    int bit_in_limb = bit_offset % 64;
    
    if (limb_idx >= Fr::LIMBS) return 0;
    
    uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;
    
    // Handle cross-limb windows
    int bits_from_first = 64 - bit_in_limb;
    if (bits_from_first < c && limb_idx + 1 < Fr::LIMBS) {
        window |= (scalar.limbs[limb_idx + 1] << bits_from_first);
    }
    
    // Mask to c bits
    return (int)(window & ((1ULL << c) - 1));
}

// =============================================================================
// Bucket Accumulation Kernel (Conflict-Free)
// =============================================================================

/**
 * @brief Conflict-free bucket accumulation
 * 
 * Each block processes one bucket. All threads in the block cooperatively
 * scan all points and accumulate those belonging to their bucket.
 * 
 * This approach guarantees no race conditions since each bucket is
 * exclusively owned by one block.
 */
__global__ void accumulate_buckets_kernel(
    G1Projective* buckets,
    const Fr* scalars,
    const G1Affine* bases,
    int msm_size,
    int c,
    int num_windows,
    int num_buckets
) {
    // Each block processes one (window, bucket) pair
    int bucket_global_idx = blockIdx.x;
    int total_buckets = num_windows * num_buckets;
    
    if (bucket_global_idx >= total_buckets) return;
    
    int window_idx = bucket_global_idx / num_buckets;
    int bucket_local = bucket_global_idx % num_buckets;
    int target_bucket = bucket_local + 1;  // buckets are 1-indexed (0 means skip)
    
    // Shared memory for parallel reduction
    extern __shared__ G1Projective shared_points[];
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each thread accumulates a portion of points
    G1Projective local_acc = G1Projective::identity();
    
    for (int i = tid; i < msm_size; i += num_threads) {
        Fr scalar = scalars[i];
        int window_val = extract_window_value(scalar, window_idx, c);
        
        if (window_val == 0) continue;
        
        // Signed digit handling: if window > 2^(c-1), use negation
        int sign = 1;
        int bucket;
        
        if (window_val > num_buckets) {
            bucket = (1 << c) - window_val;
            sign = -1;
        } else {
            bucket = window_val;
        }
        
        if (bucket == target_bucket) {
            G1Affine point = bases[i];
            if (!point.is_identity()) {
                if (sign < 0) {
                    point = point.neg();
                }
                g1_add_mixed(local_acc, local_acc, point);
            }
        }
    }
    
    // Store to shared memory
    shared_points[tid] = local_acc;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            g1_add(shared_points[tid], shared_points[tid], shared_points[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        buckets[bucket_global_idx] = shared_points[0];
    }
}

// =============================================================================
// Bucket Reduction (Triangle Sum)
// =============================================================================

/**
 * @brief Bucket reduction using triangle sum
 * 
 * Computes: sum_{i=1}^{n} i * bucket[i]
 * 
 * Equivalent to: running_sum from highest to lowest bucket,
 * accumulating running_sum into window_sum at each step.
 */
__global__ void bucket_reduction_kernel(
    G1Projective* window_results,
    const G1Projective* buckets,
    int num_windows,
    int num_buckets
) {
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (window_idx >= num_windows) return;
    
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

// =============================================================================
// Final Window Combination
// =============================================================================

/**
 * @brief Combine window results: result = sum_{w} 2^{w*c} * window_result[w]
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
 * Production-ready implementation with conflict-free accumulation.
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
        P identity = P::identity();
        if (config.are_results_on_device) {
            cudaMemcpy(result, &identity, sizeof(P), cudaMemcpyHostToDevice);
        } else {
            *result = identity;
        }
        return cudaSuccess;
    }

    cudaStream_t stream = config.stream;
    cudaError_t err;
    
    // Determine parameters
    int c = config.c > 0 ? config.c : get_optimal_c(msm_size);
    
    // Security check: prevent integer overflow and excessive memory usage
    if (c > 24) return cudaErrorInvalidValue;

    int scalar_bits = config.bitsize > 0 ? config.bitsize : S::LIMBS * 64;
    int num_windows = get_num_windows(scalar_bits, c);
    int num_buckets = (1 << (c - 1));  // Signed digit representation
    
    // Check for overflow in total_buckets
    long long total_buckets_long = (long long)num_windows * num_buckets;
    if (total_buckets_long > (long long)2147483647) return cudaErrorInvalidValue;
    
    int total_buckets = (int)total_buckets_long;
    
    // Allocate device memory
    S* d_scalars = nullptr;
    A* d_bases = nullptr;
    P* d_result = nullptr;
    G1Projective* d_buckets = nullptr;
    G1Projective* d_window_results = nullptr;
    
    // Handle input data
    if (config.are_scalars_on_device) {
        d_scalars = const_cast<S*>(scalars);
    } else {
        err = cudaMalloc(&d_scalars, msm_size * sizeof(S));
        if (err != cudaSuccess) return err;
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
    
    err = cudaMalloc(&d_buckets, total_buckets * sizeof(G1Projective));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_window_results, num_windows * sizeof(G1Projective));
    if (err != cudaSuccess) goto cleanup;
    
    if (config.are_results_on_device) {
        d_result = result;
    } else {
        err = cudaMalloc(&d_result, sizeof(P));
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Bucket accumulation: each block handles one bucket
    {
        // Determine block size based on MSM size
        int threads_per_block = 256;
        if (msm_size < 256) threads_per_block = 64;
        if (msm_size < 64) threads_per_block = 32;
        
        size_t shared_size = threads_per_block * sizeof(G1Projective);
        
        accumulate_buckets_kernel<<<total_buckets, threads_per_block, 
                                    shared_size, stream>>>(
            d_buckets,
            d_scalars,
            d_bases,
            msm_size,
            c,
            num_windows,
            num_buckets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Bucket reduction (triangle sum)
    {
        int threads = min(num_windows, 256);
        int blocks = (num_windows + threads - 1) / threads;
        
        bucket_reduction_kernel<<<blocks, threads, 0, stream>>>(
            d_window_results,
            d_buckets,
            num_windows,
            num_buckets
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Final accumulation
    {
        final_accumulation_kernel<<<1, 1, 0, stream>>>(
            d_result,
            d_window_results,
            num_windows,
            c
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
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
    if (d_buckets) cudaFree(d_buckets);
    if (d_window_results) cudaFree(d_window_results);
    if (!config.are_scalars_on_device && d_scalars) cudaFree(d_scalars);
    if (!config.are_points_on_device && d_bases) cudaFree(d_bases);
    if (!config.are_results_on_device && d_result) cudaFree(d_result);
    
    return err;
}

} // namespace msm
