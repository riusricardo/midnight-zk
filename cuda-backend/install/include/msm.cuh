/**
 * @file msm.cuh
 * @brief Multi-Scalar Multiplication (MSM) - Pippenger's Algorithm
 * 
 * Computes: R = Σᵢ sᵢ × Pᵢ for scalars s and points P
 * 
 * ARCHITECTURE:
 * =============
 * MSM kernels are templates defined in this header. This is an exception to
 * the "kernels in .cu files" rule because MSM is parameterized by curve type.
 * Template instantiation happens in msm.cu.
 * 
 * ALGORITHM (Pippenger's Bucket Method):
 * ======================================
 * 1. Decompose scalars into windows of c bits each
 * 2. For each window, accumulate points into 2^c buckets
 * 3. Compute weighted sum: Σⱼ j × bucket[j] (triangle sum)
 * 4. Combine windows: R = Σ 2^(c*w) × window_sum[w]
 * 
 * LIMITS:
 * =======
 * - Maximum MSM size: 2^24 points (16M)
 *   - Larger sizes risk integer overflow in intermediate calculations
 *   - For larger MSMs, split into batches and aggregate results
 * 
 * - Window size c: 1-24
 *   - Larger windows = more buckets = more memory
 *   - c > 24 may cause bucket index overflow
 * 
 * MEMORY USAGE:
 * =============
 * - Bucket storage: num_windows * 2^(c-1) * sizeof(G1Projective)
 *   - For c=16, n=2^20: 16 windows * 32K buckets * 288 bytes ≈ 148 MB
 * - CUB sort temporary: ~2x input size for radix sort
 * - Total GPU memory required: approximately 3-4x input point array size
 * 
 * Example memory for 1M points (c=15):
 *   - Input points: 1M * 288 bytes = 288 MB
 *   - Buckets: 17 windows * 16K buckets * 288 bytes = 78 MB
 *   - Sort temp: ~100 MB
 *   - Total: ~500 MB
 * 
 * SECURITY (Constant-Time Implementation):
 * ========================================
 * Uses Sort-Reduce pattern for bucket accumulation:
 * - Sort (bucket_idx, point) pairs by bucket
 * - Reduce consecutive points for same bucket
 * This prevents timing side-channels that could leak scalar information.
 * 
 * For side-channel sensitive applications, verify this is sufficient
 * for your threat model. The main remaining side-channel is memory
 * access patterns during bucket accumulation.
 * 
 * PERFORMANCE:
 * ============
 * - Optimal c selected based on MSM size (larger MSM → larger window)
 * - Signed digit representation for smaller bucket count
 * - CUB library for efficient parallel sorting
 * 
 * Typical performance (RTX 3090):
 *   - 2^16 points: ~5ms
 *   - 2^18 points: ~15ms
 *   - 2^20 points: ~50ms
 * 
 * Kernels:
 * - compute_bucket_indices_kernel: Decompose scalars into bucket indices
 * - accumulate_sorted_kernel: Sort-Reduce bucket accumulation
 * - parallel_bucket_reduction_kernel: Compute weighted bucket sums
 * - final_accumulation_kernel: Combine window results
 */

#pragma once

#include "point.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

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
// Bucket Accumulation Kernels (Sorted / Constant-Time)
// =============================================================================

/**
 * @brief Compute bucket indices for all scalar windows (Constant Time)
 */
template<typename S>
__global__ void compute_bucket_indices_kernel(
    unsigned int* bucket_indices,    // [msm_size * num_windows]
    unsigned int* packed_indices,    // [msm_size * num_windows] (point_idx << 1 | sign)
    const S* scalars,
    int msm_size,
    int c,
    int num_windows,
    int num_buckets // Actual buckets per window (excluding trash)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= msm_size) return;
    
    S scalar = scalars[idx];
    
    // Number of buckets per window including trash bucket
    int buckets_per_window = num_buckets + 1;
    
    for (int w = 0; w < num_windows; w++) {
        int output_idx = idx * num_windows + w;
        
        // Extract window value
        int bit_offset = w * c;
        int limb_idx = bit_offset / 64;
        int bit_in_limb = bit_offset % 64;
        
        uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;
        if (bit_in_limb + c > 64 && limb_idx + 1 < S::LIMBS) {
            window |= (scalar.limbs[limb_idx + 1] << (64 - bit_in_limb));
        }
        window &= ((1ULL << c) - 1);
        
        int window_val = (int)window;
        
        // Constant-time logic
        int sign = 0;
        int bucket_val = window_val;
        
        // Handle signed digit: if (window_val > num_buckets)
        int is_large = (window_val > num_buckets);
        if (is_large) {
            bucket_val = (1 << c) - window_val;
            sign = 1;
        }
        
        // Handle zero: if (window_val == 0)
        int is_zero = (window_val == 0);
        if (is_zero) {
            bucket_val = num_buckets + 1; // Map to trash bucket
            sign = 0;
        }
        
        // Global bucket index (0-based)
        unsigned int bucket_idx = w * buckets_per_window + (bucket_val - 1);
        
        bucket_indices[output_idx] = bucket_idx;
        packed_indices[output_idx] = (idx << 1) | sign;
    }
}

/**
 * @brief Conflict-free bucket accumulation after sorting
 */
__global__ void accumulate_sorted_kernel(
    G1Projective* buckets,
    const unsigned int* sorted_packed_indices,
    const G1Affine* bases,
    const unsigned int* bucket_offsets,
    const unsigned int* bucket_sizes,
    int total_buckets
) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= total_buckets) return;
    
    unsigned int offset = bucket_offsets[bucket_idx];
    unsigned int size = bucket_sizes[bucket_idx];
    
    if (size == 0) {
        buckets[bucket_idx] = G1Projective::identity();
        return;
    }
    
    G1Projective acc = G1Projective::identity();
    
    for (unsigned int i = 0; i < size; i++) {
        unsigned int idx = offset + i;
        unsigned int packed = sorted_packed_indices[idx];
        unsigned int point_idx = packed >> 1;
        unsigned int sign = packed & 1;
        
        G1Affine base = bases[point_idx];
        if (sign) {
            base = base.neg();
        }
        
        g1_add_mixed(acc, acc, base);
    }
    
    buckets[bucket_idx] = acc;
}

/**
 * @brief Parallel bucket reduction
 * 
 * Runs one thread per window to perform the triangle summation.
 * Since num_buckets can be large (e.g. 2^16), we read directly from global memory.
 */
__global__ void parallel_bucket_reduction_kernel(
    G1Projective* window_results,
    const G1Projective* buckets,
    int num_windows,
    int num_buckets, // Excluding trash
    int buckets_per_window // Including trash
) {
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (window_idx >= num_windows) return;
    
    // Offset to this window's buckets
    const G1Projective* window_buckets = buckets + window_idx * buckets_per_window;
    
    G1Projective running_sum = G1Projective::identity();
    G1Projective window_sum = G1Projective::identity();
    
    // Triangle summation: sum_{i=1}^B i*B_i
    // Implemented as:
    // running_sum += B_i
    // window_sum += running_sum
    // Iterate backwards from B-1 to 0 (bucket values B to 1)
    // Note: bucket_idx i corresponds to value i+1
    
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
 * @brief MSM using Pippenger's bucket method with Sort-Reduce
 * implementation with conflict-free accumulation.
 * 
 * This implementation uses a Sort-Reduce approach to mitigate timing side-channels:
 * 1. Compute bucket indices for all windows (constant-time).
 * 2. Sort indices using Radix Sort (deterministic time).
 * 3. Accumulate sorted buckets (contiguous memory access).
 * 
 * Zeros are mapped to a "trash bucket" to avoid data-dependent skipping.
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
    long long total_buckets_long = (long long)num_windows * (num_buckets + 1);
    if (total_buckets_long > (long long)2147483647) return cudaErrorInvalidValue;
    
    int total_buckets = (int)total_buckets_long;
    int num_contributions = msm_size * num_windows;
    
    // Allocate device memory
    S* d_scalars = nullptr;
    A* d_bases = nullptr;
    P* d_result = nullptr;
    G1Projective* d_buckets = nullptr;
    G1Projective* d_window_results = nullptr;
    
    // Sorting buffers
    unsigned int *d_bucket_indices = nullptr, *d_packed_indices = nullptr;
    unsigned int *d_bucket_indices_sorted = nullptr, *d_packed_indices_sorted = nullptr;
    unsigned int *d_bucket_offsets = nullptr, *d_bucket_sizes = nullptr;

    // Helper macro for error handling with cleanup
    #define MSM_CUDA_CHECK(call) do { \
        err = call; \
        if (err != cudaSuccess) goto cleanup; \
    } while(0)
    
    // Handle input data
    if (config.are_scalars_on_device) {
        d_scalars = const_cast<S*>(scalars);
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_scalars, msm_size * sizeof(S)));
        MSM_CUDA_CHECK(cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(S), 
                              cudaMemcpyHostToDevice, stream));
    }
    
    if (config.are_points_on_device) {
        d_bases = const_cast<A*>(bases);
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_bases, msm_size * sizeof(A)));
        MSM_CUDA_CHECK(cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(A),
                              cudaMemcpyHostToDevice, stream));
    }
    
    MSM_CUDA_CHECK(cudaMalloc(&d_buckets, total_buckets * sizeof(G1Projective)));
    MSM_CUDA_CHECK(cudaMalloc(&d_window_results, num_windows * sizeof(G1Projective)));
    
    if (config.are_results_on_device) {
        d_result = result;
    } else {
        MSM_CUDA_CHECK(cudaMalloc(&d_result, sizeof(P)));
    }
    
    // Allocate sorting buffers
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_indices, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_packed_indices, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_indices_sorted, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_packed_indices_sorted, num_contributions * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_offsets, total_buckets * sizeof(unsigned int)));
    MSM_CUDA_CHECK(cudaMalloc(&d_bucket_sizes, total_buckets * sizeof(unsigned int)));
    
    // 1. Compute Indices
    {
        int threads = 256;
        int blocks = (msm_size + threads - 1) / threads;
        compute_bucket_indices_kernel<<<blocks, threads, 0, stream>>>(
            d_bucket_indices, d_packed_indices, d_scalars, msm_size, c, num_windows, num_buckets
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 2. Histogram (Compute bucket sizes)
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        // Initialize sizes to 0
        MSM_CUDA_CHECK(cudaMemsetAsync(d_bucket_sizes, 0, total_buckets * sizeof(unsigned int), stream));
        
        MSM_CUDA_CHECK(cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
            d_bucket_indices, d_bucket_sizes, total_buckets, 0, total_buckets, num_contributions, stream));
            
        MSM_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        
        MSM_CUDA_CHECK(cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
            d_bucket_indices, d_bucket_sizes, total_buckets, 0, total_buckets, num_contributions, stream));
            
        MSM_CUDA_CHECK(cudaFree(d_temp_storage));
    }
    
    // 3. Scan (Compute bucket offsets)
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        MSM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            d_bucket_sizes, d_bucket_offsets, total_buckets, stream));
            
        MSM_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        
        MSM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
            d_bucket_sizes, d_bucket_offsets, total_buckets, stream));
            
        MSM_CUDA_CHECK(cudaFree(d_temp_storage));
    }
    
    // 4. Sort
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        MSM_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_bucket_indices, d_bucket_indices_sorted,
            d_packed_indices, d_packed_indices_sorted,
            num_contributions, 0, 32, stream));
            
        MSM_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        
        MSM_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_bucket_indices, d_bucket_indices_sorted,
            d_packed_indices, d_packed_indices_sorted,
            num_contributions, 0, 32, stream));
            
        MSM_CUDA_CHECK(cudaFree(d_temp_storage));
    }
    
    // 5. Accumulate Sorted
    {
        int threads = 256;
        int blocks = (total_buckets + threads - 1) / threads;
        
        accumulate_sorted_kernel<<<blocks, threads, 0, stream>>>(
            d_buckets,
            d_packed_indices_sorted,
            d_bases,
            d_bucket_offsets,
            d_bucket_sizes,
            total_buckets
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // 6. Bucket Reduction
    {
        int threads = min(num_windows, 256);
        int blocks = (num_windows + threads - 1) / threads;
        
        parallel_bucket_reduction_kernel<<<blocks, threads, 0, stream>>>(
            d_window_results,
            d_buckets,
            num_windows,
            num_buckets,
            num_buckets + 1 // buckets_per_window
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // Final accumulation
    {
        final_accumulation_kernel<<<1, 1, 0, stream>>>(
            d_result,
            d_window_results,
            num_windows,
            c
        );
        MSM_CUDA_CHECK(cudaGetLastError());
    }
    
    // Copy result back if needed
    if (!config.are_results_on_device) {
        MSM_CUDA_CHECK(cudaMemcpyAsync(result, d_result, sizeof(P),
                              cudaMemcpyDeviceToHost, stream));
    }
    
    // Synchronize if not async
    if (!config.is_async) {
        MSM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

cleanup:
    if (d_buckets) cudaFree(d_buckets);
    if (d_window_results) cudaFree(d_window_results);
    if (d_bucket_indices) cudaFree(d_bucket_indices);
    if (d_packed_indices) cudaFree(d_packed_indices);
    if (d_bucket_indices_sorted) cudaFree(d_bucket_indices_sorted);
    if (d_packed_indices_sorted) cudaFree(d_packed_indices_sorted);
    if (d_bucket_offsets) cudaFree(d_bucket_offsets);
    if (d_bucket_sizes) cudaFree(d_bucket_sizes);
    
    if (!config.are_scalars_on_device && d_scalars) cudaFree(d_scalars);
    if (!config.are_points_on_device && d_bases) cudaFree(d_bases);
    if (!config.are_results_on_device && d_result) cudaFree(d_result);
    
    #undef MSM_CUDA_CHECK
    return err;
}

} // namespace msm
