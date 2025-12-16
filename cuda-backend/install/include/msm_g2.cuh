/**
 * @file msm_g2.cuh
 * @brief G2 Multi-Scalar Multiplication using Pippenger's bucket method
 * 
 * Separate implementation for G2 as it uses Fq2 extension field.
 */

#pragma once

#include "point.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>

namespace msm_g2 {

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// Scalar Window Extraction (same as G1)
// =============================================================================

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
    
    int bits_from_first = 64 - bit_in_limb;
    if (bits_from_first < c && limb_idx + 1 < Fr::LIMBS) {
        window |= (scalar.limbs[limb_idx + 1] << bits_from_first);
    }
    
    return (int)(window & ((1ULL << c) - 1));
}

__host__ __device__ inline int get_optimal_c(int msm_size) {
    if (msm_size <= 1) return 1;
    
    int log_size = 0;
    int temp = msm_size;
    while (temp > 1) {
        temp >>= 1;
        log_size++;
    }
    
    // G2 operations are more expensive, so use slightly smaller windows
    if (log_size <= 8)  return 6;
    if (log_size <= 10) return 7;
    if (log_size <= 12) return 9;
    if (log_size <= 14) return 11;
    if (log_size <= 16) return 12;
    if (log_size <= 18) return 13;
    if (log_size <= 20) return 14;
    return 15;
}

__host__ __device__ inline int get_num_windows(int scalar_bits, int c) {
    return (scalar_bits + c - 1) / c;
}

// =============================================================================
// G2 Bucket Accumulation Kernel
// =============================================================================

__global__ void accumulate_g2_buckets_kernel(
    G2Projective* buckets,
    const Fr* scalars,
    const G2Affine* bases,
    int msm_size,
    int c,
    int num_windows,
    int num_buckets
) {
    int bucket_global_idx = blockIdx.x;
    int total_buckets = num_windows * num_buckets;
    
    if (bucket_global_idx >= total_buckets) return;
    
    int window_idx = bucket_global_idx / num_buckets;
    int bucket_local = bucket_global_idx % num_buckets;
    int target_bucket = bucket_local + 1;
    
    extern __shared__ G2Projective shared_g2_points[];
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    G2Projective local_acc = G2Projective::identity();
    
    for (int i = tid; i < msm_size; i += num_threads) {
        Fr scalar = scalars[i];
        int window_val = extract_window_value(scalar, window_idx, c);
        
        if (window_val == 0) continue;
        
        int sign = 1;
        int bucket;
        
        if (window_val > num_buckets) {
            bucket = (1 << c) - window_val;
            sign = -1;
        } else {
            bucket = window_val;
        }
        
        if (bucket == target_bucket) {
            G2Affine point = bases[i];
            if (!point.is_identity()) {
                if (sign < 0) {
                    point = point.neg();
                }
                g2_add_mixed(local_acc, local_acc, point);
            }
        }
    }
    
    shared_g2_points[tid] = local_acc;
    __syncthreads();
    
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            g2_add(shared_g2_points[tid], shared_g2_points[tid], shared_g2_points[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        buckets[bucket_global_idx] = shared_g2_points[0];
    }
}

// =============================================================================
// G2 Bucket Reduction
// =============================================================================

__global__ void g2_bucket_reduction_kernel(
    G2Projective* window_results,
    const G2Projective* buckets,
    int num_windows,
    int num_buckets
) {
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (window_idx >= num_windows) return;
    
    const G2Projective* window_buckets = buckets + window_idx * num_buckets;
    
    G2Projective running_sum = G2Projective::identity();
    G2Projective window_sum = G2Projective::identity();
    
    for (int i = num_buckets - 1; i >= 0; i--) {
        g2_add(running_sum, running_sum, window_buckets[i]);
        g2_add(window_sum, window_sum, running_sum);
    }
    
    window_results[window_idx] = window_sum;
}

// =============================================================================
// G2 Final Accumulation
// =============================================================================

__global__ void g2_final_accumulation_kernel(
    G2Projective* result,
    const G2Projective* window_results,
    int num_windows,
    int c
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    G2Projective acc = window_results[num_windows - 1];
    
    for (int w = num_windows - 2; w >= 0; w--) {
        for (int i = 0; i < c; i++) {
            g2_double(acc, acc);
        }
        g2_add(acc, acc, window_results[w]);
    }
    
    *result = acc;
}

// =============================================================================
// G2 MSM Entry Point
// =============================================================================

inline cudaError_t g2_msm_cuda(
    const Fr* scalars,
    const G2Affine* bases,
    int msm_size,
    const MSMConfig& config,
    G2Projective* result
) {
    if (msm_size == 0) {
        G2Projective identity = G2Projective::identity();
        if (config.are_results_on_device) {
            cudaMemcpy(result, &identity, sizeof(G2Projective), cudaMemcpyHostToDevice);
        } else {
            *result = identity;
        }
        return cudaSuccess;
    }

    cudaStream_t stream = config.stream;
    cudaError_t err;
    
    int c = config.c > 0 ? config.c : get_optimal_c(msm_size);
    int scalar_bits = config.bitsize > 0 ? config.bitsize : Fr::LIMBS * 64;
    int num_windows = get_num_windows(scalar_bits, c);
    int num_buckets = (1 << (c - 1));
    int total_buckets = num_windows * num_buckets;
    
    Fr* d_scalars = nullptr;
    G2Affine* d_bases = nullptr;
    G2Projective* d_result = nullptr;
    G2Projective* d_buckets = nullptr;
    G2Projective* d_window_results = nullptr;

    // Helper macro for error handling with cleanup
    #define MSM_CUDA_CHECK(call) do { \
        err = call; \
        if (err != cudaSuccess) goto cleanup; \
    } while(0)
    
    if (config.are_scalars_on_device) {
        d_scalars = const_cast<Fr*>(scalars);
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_scalars, msm_size * sizeof(Fr)));
        MSM_CUDA_CHECK(cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(Fr), 
                              cudaMemcpyHostToDevice, stream));
    }
    
    if (config.are_points_on_device) {
        d_bases = const_cast<G2Affine*>(bases);
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_bases, msm_size * sizeof(G2Affine)));
        MSM_CUDA_CHECK(cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(G2Affine),
                              cudaMemcpyHostToDevice, stream));
    }
    
    MSM_CUDA_CHECK(cudaMalloc(&d_buckets, total_buckets * sizeof(G2Projective)));
    MSM_CUDA_CHECK(cudaMalloc(&d_window_results, num_windows * sizeof(G2Projective)));
    
    if (config.are_results_on_device) {
        d_result = result;
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_result, sizeof(G2Projective)));
    }
    
    // Bucket accumulation
    {
        int threads_per_block = 128;  // Fewer threads for larger G2 points
        if (msm_size < 128) threads_per_block = 64;
        if (msm_size < 64) threads_per_block = 32;
        
        size_t shared_size = threads_per_block * sizeof(G2Projective);
        
        accumulate_g2_buckets_kernel<<<total_buckets, threads_per_block, 
                                       shared_size, stream>>>(
            d_buckets,
            d_scalars,
            d_bases,
            msm_size,
            c,
            num_windows,
            num_buckets
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // Bucket reduction
    {
        int threads = min(num_windows, 256);
        int blocks = (num_windows + threads - 1) / threads;
        
        g2_bucket_reduction_kernel<<<blocks, threads, 0, stream>>>(
            d_window_results,
            d_buckets,
            num_windows,
            num_buckets
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // Final accumulation
    {
        g2_final_accumulation_kernel<<<1, 1, 0, stream>>>(
            d_result,
            d_window_results,
            num_windows,
            c
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    if (!config.are_results_on_device) {
        MSM_CUDA_CHECK(cudaMemcpyAsync(result, d_result, sizeof(G2Projective),
                              cudaMemcpyDeviceToHost, stream));
    }
    
    if (!config.is_async) {
        MSM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

cleanup:
    if (d_buckets) cudaFree(d_buckets);
    if (d_window_results) cudaFree(d_window_results);
    if (!config.are_scalars_on_device && d_scalars) cudaFree(d_scalars);
    if (!config.are_points_on_device && d_bases) cudaFree(d_bases);
    if (!config.are_results_on_device && d_result) cudaFree(d_result);
    
    #undef MSM_CUDA_CHECK
    return err;
}

} // namespace msm_g2
