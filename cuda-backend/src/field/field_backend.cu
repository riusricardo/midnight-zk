/**
 * @file field_backend.cu
 * @brief NTT and Field Backend Implementation
 * 
 * This file contains the complete NTT implementation for the BLS12-381 scalar field.
 * 
 * ARCHITECTURE:
 * =============
 * CUDA static libraries require kernels to be defined in the same compilation unit
 * that calls them. This file is self-contained with:
 * 
 * 1. NTT Kernels (defined here, called here):
 *    - ntt_bit_reverse_kernel: Bit-reversal permutation
 *    - ntt_butterfly_kernel: Single Radix-2 butterfly stage
 *    - ntt_scale_kernel: Scaling for inverse NTT
 *    - ntt_shared_memory_kernel: Optimized for small sizes (≤1024)
 *    - intt_shared_memory_kernel: Inverse version of above
 *    - ntt_butterfly_fused_2stage_kernel: Fused 2-stage for larger sizes
 *    - ntt_radix4_kernel: True Radix-4 butterfly (alternative optimization)
 * 
 * 2. Domain Management:
 *    - Static storage for Domain<Fr> instances
 *    - Twiddle factor precomputation kernels
 * 
 * 3. API Functions:
 *    - ntt_cuda: Main NTT entry point
 *    - coset_ntt_cuda: Coset NTT for polynomial evaluation
 *    - init_domain_cuda: Initialize twiddle factors
 *    - release_domain_cuda: Free resources
 * 
 * OPTIMIZATIONS:
 * ==============
 * - Small NTT (≤1024): Uses shared memory kernel - entire transform in SRAM
 * - Large NTT (>1024): Uses fused 2-stage butterfly - 50% fewer kernel launches
 * - All operations use Montgomery form for efficient modular arithmetic
 */

#include "field.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>
#include <iostream>

using namespace bls12_381;
using namespace ntt;

// =============================================================================
// Global NTT Domain Registry - Storage Definitions
// =============================================================================

// Define static members for Domain<Fr> - must be outside namespace
namespace ntt {
    template<>
    Domain<Fr>* Domain<Fr>::domains[MAX_LOG_DOMAIN_SIZE] = {nullptr};

    template<>
    std::mutex Domain<Fr>::domains_mutex{};
}

template<>
Domain<Fr>* Domain<Fr>::get_domain(int log_size) {
    if (log_size >= MAX_LOG_DOMAIN_SIZE) return nullptr;
    std::lock_guard<std::mutex> lock(domains_mutex);
    return domains[log_size];
}

template<>
void Domain<Fr>::set_domain(int log_size, Domain<Fr>* domain) {
    if (log_size >= MAX_LOG_DOMAIN_SIZE) return;
    std::lock_guard<std::mutex> lock(domains_mutex);
    domains[log_size] = domain;
}

// =============================================================================
// NTT Core Implementation - Kernels and Host Functions
// =============================================================================

namespace ntt {

// =============================================================================
// NTT Kernels (must be in same compilation unit for proper linking)
// =============================================================================

/**
 * @brief Bit-reverse permutation kernel for Fr
 */
__global__ void ntt_bit_reverse_kernel(
    Fr* output,
    const Fr* input,
    int n,
    int log_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Compute bit-reversed index
    unsigned int rev = 0;
    unsigned int temp = idx;
    for (int i = 0; i < log_n; i++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }
    
    output[rev] = input[idx];
}

/**
 * @brief Single butterfly stage kernel for Fr
 */
__global__ void ntt_butterfly_kernel(
    Fr* data,
    const Fr* twiddles,
    int n,
    int m,           // Group size = 2^(stage+1)
    int half_m       // Half group size = 2^stage
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_butterflies = n / 2;
    
    if (idx >= num_butterflies) return;
    
    // Determine which butterfly group and position within group
    int group = idx / half_m;
    int pos = idx % half_m;
    
    // Indices of the two elements in the butterfly
    int i = group * m + pos;
    int j = i + half_m;
    
    // Twiddle factor index
    int twiddle_idx = pos * (n / m);
    
    Fr twiddle = twiddles[twiddle_idx];
    
    // Butterfly operation
    Fr u = data[i];
    Fr v = data[j] * twiddle;
    
    data[i] = u + v;
    data[j] = u - v;
}

/**
 * @brief Scale kernel for Fr (inverse NTT)
 */
__global__ void ntt_scale_kernel(
    Fr* output,
    const Fr* input,
    Fr scale_factor,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    output[idx] = input[idx] * scale_factor;
}

// =============================================================================
// Optimized NTT Kernels
// =============================================================================

/**
 * @brief Shared memory NTT kernel for small transforms
 * 
 * Processes entire NTT in shared memory when size <= 1024.
 * This dramatically reduces global memory traffic - only 2 accesses per element
 * instead of 2*log_size accesses with the standard approach.
 * 
 * @param data Input/output array (in-place transform)
 * @param twiddles Precomputed twiddle factors
 * @param size NTT size (must be power of 2, <= 1024)
 * @param log_size log2(size)
 */
__global__ void ntt_shared_memory_kernel(
    Fr* data,
    const Fr* twiddles,
    int size,
    int log_size
) {
    extern __shared__ Fr sdata[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * size;
    
    // Load data into shared memory with bit-reversal permutation
    if (tid < size) {
        unsigned int rev = 0;
        unsigned int n = tid;
        for (int i = 0; i < log_size; i++) {
            rev = (rev << 1) | (n & 1);
            n >>= 1;
        }
        sdata[tid] = data[block_offset + rev];
    }
    __syncthreads();
    
    // All butterfly stages in shared memory
    for (int s = 1; s <= log_size; s++) {
        int m = 1 << s;
        int half_m = m / 2;
        
        if (tid < size / 2) {
            int group = tid / half_m;
            int pos = tid % half_m;
            
            int i0 = group * m + pos;
            int i1 = i0 + half_m;
            
            int twiddle_idx = pos * (size / m);
            Fr omega = twiddles[twiddle_idx];
            
            Fr u = sdata[i0];
            Fr t = sdata[i1] * omega;
            
            sdata[i0] = u + t;
            sdata[i1] = u - t;
        }
        __syncthreads();
    }
    
    // Write back to global memory
    if (tid < size) {
        data[block_offset + tid] = sdata[tid];
    }
}

/**
 * @brief Inverse shared memory NTT kernel
 * 
 * Like ntt_shared_memory_kernel but for inverse transform.
 * Uses inverse twiddles and scales by 1/n at the end.
 */
__global__ void intt_shared_memory_kernel(
    Fr* data,
    const Fr* inv_twiddles,
    Fr scale_factor,
    int size,
    int log_size
) {
    extern __shared__ Fr sdata[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * size;
    
    // Load data into shared memory (no bit-reversal for inverse)
    if (tid < size) {
        sdata[tid] = data[block_offset + tid];
    }
    __syncthreads();
    
    // Butterfly stages in reverse order
    for (int s = log_size; s >= 1; s--) {
        int m = 1 << s;
        int half_m = m / 2;
        
        if (tid < size / 2) {
            int group = tid / half_m;
            int pos = tid % half_m;
            
            int i0 = group * m + pos;
            int i1 = i0 + half_m;
            
            int twiddle_idx = pos * (size / m);
            Fr omega_inv = inv_twiddles[twiddle_idx];
            
            Fr u = sdata[i0];
            Fr t = sdata[i1] * omega_inv;
            
            sdata[i0] = u + t;
            sdata[i1] = u - t;
        }
        __syncthreads();
    }
    
    // Bit-reverse and scale, write back to global memory
    if (tid < size) {
        unsigned int rev = 0;
        unsigned int n = tid;
        for (int i = 0; i < log_size; i++) {
            rev = (rev << 1) | (n & 1);
            n >>= 1;
        }
        data[block_offset + rev] = sdata[tid] * scale_factor;
    }
}

/**
 * @brief Fused 2-stage butterfly kernel
 * 
 * Performs two consecutive butterfly stages in one kernel launch.
 * Reduces kernel launch overhead by 50% and halves global memory traffic
 * compared to running two separate butterfly kernels.
 * 
 * Each thread processes 4 elements through 2 stages.
 * 
 * @param data Input/output array
 * @param twiddles Precomputed twiddle factors
 * @param n Total NTT size
 * @param stride First stage half_m value (stride of first stage)
 */
__global__ void ntt_butterfly_fused_2stage_kernel(
    Fr* data,
    const Fr* twiddles,
    int n,
    int stride  // half_m of the first stage being processed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_quads = n / 4;
    
    if (idx >= num_quads) return;
    
    // Each thread handles a group of 4 consecutive elements
    // Group size for fused 2 stages = 4 * stride
    int quad_m = 4 * stride;
    int group = idx / stride;
    int pos = idx % stride;
    
    int base = group * quad_m + pos;
    
    int i0 = base;
    int i1 = base + stride;
    int i2 = base + 2 * stride;
    int i3 = base + 3 * stride;
    
    // Load 4 elements
    Fr x0 = data[i0];
    Fr x1 = data[i1];
    Fr x2 = data[i2];
    Fr x3 = data[i3];
    
    // ---------------------------------------------------------
    // Stage 1 (group size = 2*stride, half_m = stride)
    // ---------------------------------------------------------
    // Pairs: (x0, x1) and (x2, x3)
    int tw_idx_1 = pos * (n / (2 * stride));
    Fr w1 = twiddles[tw_idx_1];
    
    // Butterfly on (x0, x1)
    Fr v1 = x1 * w1;
    Fr u1 = x0;
    x0 = u1 + v1;
    x1 = u1 - v1;
    
    // Butterfly on (x2, x3) with same twiddle
    Fr v2 = x3 * w1;
    Fr u2 = x2;
    x2 = u2 + v2;
    x3 = u2 - v2;
    
    // ---------------------------------------------------------
    // Stage 2 (group size = 4*stride, half_m = 2*stride)
    // ---------------------------------------------------------
    // Pairs: (x0, x2) and (x1, x3)
    int tw_idx_2a = pos * (n / (4 * stride));
    Fr w2a = twiddles[tw_idx_2a];
    
    // Butterfly on (x0, x2)
    Fr v3 = x2 * w2a;
    Fr u3 = x0;
    x0 = u3 + v3;
    x2 = u3 - v3;
    
    // Butterfly on (x1, x3)
    int tw_idx_2b = (pos + stride) * (n / (4 * stride));
    Fr w2b = twiddles[tw_idx_2b];
    
    Fr v4 = x3 * w2b;
    Fr u4 = x1;
    x1 = u4 + v4;
    x3 = u4 - v4;
    
    // Store 4 elements
    data[i0] = x0;
    data[i1] = x1;
    data[i2] = x2;
    data[i3] = x3;
}

/**
 * @brief Radix-4 butterfly kernel for mixed-radix NTT
 * 
 * Processes 4 elements using the Radix-4 DFT matrix, which combines
 * 2 stages of Radix-2 butterflies mathematically.
 * 
 * Radix-4 DFT: [1  1  1  1 ]   [a0]
 *              [1 -i -1  i ] * [a1]
 *              [1 -1  1 -1 ]   [a2]
 *              [1  i -1 -i ]   [a3]
 * 
 * @param data Input/output array
 * @param twiddles Precomputed twiddle factors (includes imaginary unit at size/4)
 * @param size Total NTT size
 * @param stride Stride between elements in a Radix-4 group
 */
__global__ void ntt_radix4_kernel(
    Fr* data,
    const Fr* twiddles,
    int size,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = size / 4;
    
    if (idx >= num_groups) return;
    
    int group = idx / stride;
    int pos = idx % stride;
    
    int base = group * stride * 4 + pos;
    
    // Load 4 elements
    Fr a0 = data[base];
    Fr a1 = data[base + stride];
    Fr a2 = data[base + 2 * stride];
    Fr a3 = data[base + 3 * stride];
    
    // Twiddle factors
    int tw_stride = size / (4 * stride);
    Fr w1 = twiddles[pos * tw_stride];
    Fr w2 = twiddles[2 * pos * tw_stride];
    Fr w3 = twiddles[3 * pos * tw_stride];
    
    // Apply twiddles to a1, a2, a3
    a1 = a1 * w1;
    a2 = a2 * w2;
    a3 = a3 * w3;
    
    // Radix-4 butterfly (standard form)
    Fr b0 = a0 + a2;
    Fr b1 = a0 - a2;
    Fr b2 = a1 + a3;
    Fr b3 = a1 - a3;
    
    // Get i = omega^(n/4), the 4th root of unity
    Fr i_val = twiddles[size / 4];
    b3 = b3 * i_val;
    
    // Final combination
    data[base] = b0 + b2;
    data[base + stride] = b1 + b3;
    data[base + 2 * stride] = b0 - b2;
    data[base + 3 * stride] = b1 - b3;
}

/**
 * @brief Forward NTT implementation with optimizations
 * 
 * Uses shared memory kernel for small sizes, fused 2-stage butterflies for larger sizes.
 */
template<typename F>
eIcicleError ntt_forward_impl(
    const F* input,
    int size,
    const NTTConfig& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    // Get domain
    Domain<F>* domain = Domain<F>::get_domain(log_size);
    if (!domain) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Validate domain is fully initialized (twiddles allocated)
    if (!domain->twiddles || !domain->inv_twiddles) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Allocate working buffer
    F* d_data = nullptr;
    bool need_alloc = !config.are_inputs_on_device;
    
    if (need_alloc) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyHostToDevice));
    } else if (input != output) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyDeviceToDevice));
    } else {
        d_data = const_cast<F*>(input);
    }
    
    const int threads = 256;
    
    // =========================================================================
    // OPTIMIZATION: Use shared memory kernel for small NTTs (size <= 1024)
    // =========================================================================
    
    // First, check if shared memory is available for small NTT optimization
    bool use_shared_mem = false;
    if (size <= 1024) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        size_t shared_mem_size = size * sizeof(F);
        // Use shared memory only if it fits in device limits
        if (shared_mem_size <= prop.sharedMemPerBlock) {
            use_shared_mem = true;
        }
    }
    
    if (use_shared_mem) {
        // Shared memory approach: single kernel does bit-reversal + all butterflies
        size_t shared_mem_size = size * sizeof(F);
        ntt_shared_memory_kernel<<<1, size, shared_mem_size>>>(
            d_data, domain->twiddles, size, log_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Standard approach for larger sizes with optimizations
        
        // Bit reversal
        F* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(F)));
        
        int blocks = (size + threads - 1) / threads;
        ntt_bit_reverse_kernel<<<blocks, threads>>>(d_temp, d_data, size, log_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(d_data, d_temp, size * sizeof(F), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(d_temp));
        
        // =====================================================================
        // OPTIMIZATION: Use fused 2-stage butterflies where possible
        // =====================================================================
        int s = 1;
        while (s + 1 <= log_size) {
            // Fuse stages s and s+1
            // stride = half_m of stage s = 2^(s-1)
            int stride = 1 << (s - 1);
            
            blocks = (size / 4 + threads - 1) / threads;
            ntt_butterfly_fused_2stage_kernel<<<blocks, threads>>>(
                d_data, domain->twiddles, size, stride
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            s += 2;  // Advance by 2 stages
        }
        
        // Handle remaining stage if log_size is odd
        if (s == log_size) {
            int m = 1 << s;
            int half_m = m / 2;
            
            blocks = (size / 2 + threads - 1) / threads;
            ntt_butterfly_kernel<<<blocks, threads>>>(
                d_data, domain->twiddles, size, m, half_m
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    // Copy output
    if (config.are_outputs_on_device) {
        if (d_data != output) {
            CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToDevice));
        }
    } else {
        CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToHost));
    }
    
    if (need_alloc || (input != output && config.are_inputs_on_device)) {
        if (d_data != input) CUDA_CHECK(cudaFree(d_data));
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Inverse NTT implementation with optimizations
 * 
 * Uses shared memory kernel for small sizes, fused 2-stage butterflies for larger sizes.
 */
template<typename F>
eIcicleError ntt_inverse_impl(
    const F* input,
    int size,
    const NTTConfig& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    Domain<F>* domain = Domain<F>::get_domain(log_size);
    if (!domain) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Validate domain is fully initialized (twiddles allocated)
    if (!domain->twiddles || !domain->inv_twiddles) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Allocate working buffer
    F* d_data = nullptr;
    bool need_alloc = !config.are_inputs_on_device;
    
    if (need_alloc) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyHostToDevice));
    } else if (input != output) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(F)));
        CUDA_CHECK(cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyDeviceToDevice));
    } else {
        d_data = const_cast<F*>(input);
    }
    
    const int threads = 256;
    
    // =========================================================================
    // OPTIMIZATION: Use shared memory kernel for small NTTs (size <= 1024)
    // =========================================================================
    
    // First, check if shared memory is available for small NTT optimization
    bool use_shared_mem = false;
    if (size <= 1024) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        size_t shared_mem_size = size * sizeof(F);
        // Use shared memory only if it fits in device limits
        if (shared_mem_size <= prop.sharedMemPerBlock) {
            use_shared_mem = true;
        }
    }
    
    if (use_shared_mem) {
        // Shared memory approach: single kernel does all butterflies + bit-reversal + scaling
        size_t shared_mem_size = size * sizeof(F);
        intt_shared_memory_kernel<<<1, size, shared_mem_size>>>(
            d_data, domain->inv_twiddles, domain->domain_size_inv, size, log_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Standard approach with fused 2-stage optimization
        
        // =====================================================================
        // OPTIMIZATION: Use fused 2-stage butterflies in reverse order
        // =====================================================================
        // For inverse NTT, we go from stage log_size down to 1
        // Fuse pairs: (log_size, log_size-1), (log_size-2, log_size-3), ...
        
        int s = log_size;
        while (s - 1 >= 1) {
            // Fuse stages s and s-1 (in reverse order)
            // For the fused kernel, we need stride = half_m of the LOWER stage (s-1)
            // half_m of stage s-1 = 2^(s-2)
            int stride = 1 << (s - 2);
            
            int blocks = (size / 4 + threads - 1) / threads;
            ntt_butterfly_fused_2stage_kernel<<<blocks, threads>>>(
                d_data, domain->inv_twiddles, size, stride
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            s -= 2;  // Go back by 2 stages
        }
        
        // Handle remaining stage if log_size is odd
        if (s == 1) {
            int m = 1 << s;
            int half_m = m / 2;
            
            int blocks = (size / 2 + threads - 1) / threads;
            ntt_butterfly_kernel<<<blocks, threads>>>(
                d_data, domain->inv_twiddles, size, m, half_m
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Bit reversal
        F* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(F)));
        
        int blocks = (size + threads - 1) / threads;
        ntt_bit_reverse_kernel<<<blocks, threads>>>(d_temp, d_data, size, log_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Scale by 1/n
        ntt_scale_kernel<<<blocks, threads>>>(
            d_data, d_temp, domain->domain_size_inv, size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaFree(d_temp));
    }
    
    // Copy output
    if (config.are_outputs_on_device) {
        if (d_data != output) {
            CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToDevice));
        }
    } else {
        CUDA_CHECK(cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToHost));
    }
    
    if (need_alloc || (input != output && config.are_inputs_on_device)) {
        if (d_data != input) CUDA_CHECK(cudaFree(d_data));
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Main NTT entry point
 */
template<typename F>
eIcicleError ntt_cuda_impl(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    F* output
) {
    if (direction == NTTDir::kForward) {
        return ntt_forward_impl(input, size, config, output);
    } else {
        return ntt_inverse_impl(input, size, config, output);
    }
}

/**
 * @brief Kernel to compute field inverse (single element)
 */
template<typename F>
__global__ void field_inv_kernel(F* out, const F* in) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        field_inv(*out, *in);
    }
}

/**
 * @brief Coset NTT implementation
 * 
 * Forward coset NTT: multiply by g^i, then NTT
 * Inverse coset NTT: INTT, then divide by g^i
 */
template<typename F>
eIcicleError coset_ntt_cuda_impl(
    const F* input,
    int size,
    NTTDir direction,
    const F& coset_gen,
    const NTTConfig& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Allocate temporary buffer
    F* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(F)));
    
    // Copy coset generator to device
    F* d_coset_gen = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coset_gen, sizeof(F)));
    CUDA_CHECK(cudaMemcpy(d_coset_gen, &coset_gen, sizeof(F), cudaMemcpyHostToDevice));
    
    if (direction == NTTDir::kForward) {
        // Step 1: Multiply by coset powers
        F* d_input = nullptr;
        bool need_alloc_input = !config.are_inputs_on_device;
        
        if (need_alloc_input) {
            CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(F)));
            CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(F), cudaMemcpyHostToDevice));
        } else {
            d_input = const_cast<F*>(input);
        }
        
        coset_mul_kernel<<<blocks, threads, 0, stream>>>(d_temp, d_input, coset_gen, size);
        
        // Check for kernel launch failure
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            if (need_alloc_input) cudaFree(d_input);
            cudaFree(d_temp);
            cudaFree(d_coset_gen);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        kernelErr = cudaStreamSynchronize(stream);
        if (kernelErr != cudaSuccess) {
            if (need_alloc_input) cudaFree(d_input);
            cudaFree(d_temp);
            cudaFree(d_coset_gen);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        if (need_alloc_input) {
            CUDA_CHECK(cudaFree(d_input));
        }
        
        // Step 2: Apply regular NTT
        NTTConfig modified_config = config;
        modified_config.are_inputs_on_device = true;
        
        eIcicleError err = ntt_forward_impl(d_temp, size, modified_config, output);
        
        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaFree(d_coset_gen));
        
        return err;
    } else {
        // Inverse coset NTT
        // Step 1: Apply inverse NTT
        NTTConfig modified_config = config;
        modified_config.are_outputs_on_device = true;
        
        eIcicleError err = ntt_inverse_impl(input, size, modified_config, d_temp);
        if (err != eIcicleError::SUCCESS) {
            CUDA_CHECK(cudaFree(d_temp));
            CUDA_CHECK(cudaFree(d_coset_gen));
            return err;
        }
        
        // Step 2: Divide by coset powers (multiply by g^(-i))
        // Compute g^(-1) on device
        F* d_coset_gen_inv = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coset_gen_inv, sizeof(F)));
        field_inv_kernel<<<1, 1, 0, stream>>>(d_coset_gen_inv, d_coset_gen);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Copy inverse back to host for kernel
        F coset_gen_inv;
        CUDA_CHECK(cudaMemcpy(&coset_gen_inv, d_coset_gen_inv, sizeof(F), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_coset_gen_inv));
        
        F* d_output = nullptr;
        bool need_alloc_output = !config.are_outputs_on_device;
        
        if (need_alloc_output) {
            CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(F)));
        } else {
            d_output = output;
        }
        
        coset_div_kernel<<<blocks, threads, 0, stream>>>(d_output, d_temp, coset_gen_inv, size);
        
        // Check for kernel launch failure
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            if (need_alloc_output) cudaFree(d_output);
            cudaFree(d_temp);
            cudaFree(d_coset_gen);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        kernelErr = cudaStreamSynchronize(stream);
        if (kernelErr != cudaSuccess) {
            if (need_alloc_output) cudaFree(d_output);
            cudaFree(d_temp);
            cudaFree(d_coset_gen);
            return eIcicleError::UNKNOWN_ERROR;
        }
        
        if (need_alloc_output) {
            CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(F), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_output));
        }
        
        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaFree(d_coset_gen));
        
        return eIcicleError::SUCCESS;
    }
}

// Explicit instantiations
template eIcicleError ntt_cuda_impl<Fr>(const Fr*, int, NTTDir, const NTTConfig&, Fr*);
template eIcicleError coset_ntt_cuda_impl<Fr>(const Fr*, int, NTTDir, const Fr&, const NTTConfig&, Fr*);

} // namespace ntt

// =============================================================================
// NTT Domain Management
// =============================================================================

namespace ntt {

// Kernel to compute twiddle factors on GPU
// Note: one_val is passed explicitly to avoid __constant__ memory issues across shared library boundaries
__global__ void compute_twiddles_kernel(
    Fr* twiddles,
    Fr* inv_twiddles,
    Fr omega,
    Fr omega_inv,
    Fr one_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Compute omega^idx using repeated squaring
    Fr pow = one_val;
    Fr base = omega;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    twiddles[idx] = pow;
    
    // Compute omega_inv^idx
    pow = one_val;
    base = omega_inv;
    exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    inv_twiddles[idx] = pow;
}

// Kernel to compute omega^(2^k)
__global__ void square_field_kernel(Fr* result, const Fr* input, int iterations) {
    Fr val = *input;
    for (int i = 0; i < iterations; i++) {
        val = val * val;
    }
    *result = val;
}

// Kernel to compute inverse and size_inv
__global__ void compute_inv_kernel(Fr* omega_inv, Fr* size_inv, const Fr* omega, int size) {
    Fr inv;
    field_inv(inv, *omega);
    *omega_inv = inv;
    
    Fr size_field = Fr::from_int((uint64_t)size);
    field_inv(*size_inv, size_field);
}

/**
 * @brief Initialize NTT domain with given root of unity
 */
template<typename F>
eIcicleError init_domain_cuda_impl(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
) {
    // Determine max log size from root of unity order
    // For BLS12-381 Fr, max is 2^32 (root has order 2^32)
    int max_log_size = 32;  // Default for BLS12-381
    
    if (config.max_log_size > 0 && config.max_log_size < max_log_size) {
        max_log_size = config.max_log_size;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    // Copy root_of_unity to device
    F* d_root;
    CUDA_CHECK(cudaMalloc(&d_root, sizeof(F)));
    CUDA_CHECK(cudaMemcpy(d_root, &root_of_unity, sizeof(F), cudaMemcpyHostToDevice));
    
    // Precompute twiddle factors for each domain size
    for (int log_size = 1; log_size <= max_log_size && log_size < MAX_LOG_DOMAIN_SIZE; log_size++) {
        int size = 1 << log_size;
        
        // Allocate domain
        Domain<F>* domain = new Domain<F>();
        domain->log_size = log_size;
        domain->size = size;
        
        // Allocate twiddle factor arrays
        CUDA_CHECK(cudaMalloc(&domain->twiddles, size * sizeof(F)));
        CUDA_CHECK(cudaMalloc(&domain->inv_twiddles, size * sizeof(F)));
        
        // Compute omega = root_of_unity^(2^(max_log - log_size)) on GPU
        F* d_omega;
        F* d_omega_inv;
        F* d_size_inv;
        CUDA_CHECK(cudaMalloc(&d_omega, sizeof(F)));
        CUDA_CHECK(cudaMalloc(&d_omega_inv, sizeof(F)));
        CUDA_CHECK(cudaMalloc(&d_size_inv, sizeof(F)));
        
        int squarings = max_log_size - log_size;
        square_field_kernel<<<1, 1, 0, stream>>>(d_omega, d_root, squarings);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Compute inverse and size_inv
        compute_inv_kernel<<<1, 1, 0, stream>>>(d_omega_inv, d_size_inv, d_omega, size);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Copy omega values back to host to pass to twiddle kernel
        F h_omega, h_omega_inv;
        CUDA_CHECK(cudaMemcpy(&h_omega, d_omega, sizeof(F), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_omega_inv, d_omega_inv, sizeof(F), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&domain->domain_size_inv, d_size_inv, sizeof(F), cudaMemcpyDeviceToHost));
        
        // Get "one" value from host to avoid __constant__ memory issues
        F h_one = F::one_host();
        
        // Compute all twiddles in parallel on GPU
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;
        compute_twiddles_kernel<<<blocks, threads, 0, stream>>>(
            domain->twiddles, domain->inv_twiddles, h_omega, h_omega_inv, h_one, size
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        CUDA_CHECK(cudaFree(d_omega));
        CUDA_CHECK(cudaFree(d_omega_inv));
        CUDA_CHECK(cudaFree(d_size_inv));
        
        // Register domain using the accessor function
        Domain<F>::set_domain(log_size, domain);
    }
    
    CUDA_CHECK(cudaFree(d_root));
    
    return eIcicleError::SUCCESS;
}

template eIcicleError init_domain_cuda_impl<Fr>(const Fr&, const NTTInitDomainConfig&);

/**
 * @brief Release NTT domain resources
 * 
 * Uses a two-phase approach to avoid holding mutex during CUDA operations:
 * 1. Collect pointers under lock and clear the domain array
 * 2. Free resources outside the lock to avoid blocking other threads
 */
template<typename F>
eIcicleError release_domain_cuda_impl() {
    // Collect pointers under lock, then free outside lock
    Domain<F>* to_delete[MAX_LOG_DOMAIN_SIZE] = {nullptr};
    
    {
        std::lock_guard<std::mutex> lock(Domain<F>::domains_mutex);
        for (int i = 0; i < MAX_LOG_DOMAIN_SIZE; i++) {
            to_delete[i] = Domain<F>::domains[i];
            Domain<F>::domains[i] = nullptr;
        }
    }
    
    // Free outside lock to avoid blocking other threads
    for (int i = 0; i < MAX_LOG_DOMAIN_SIZE; i++) {
        Domain<F>* domain = to_delete[i];
        if (domain) {
            if (domain->twiddles) CUDA_CHECK(cudaFree(domain->twiddles));
            if (domain->inv_twiddles) CUDA_CHECK(cudaFree(domain->inv_twiddles));
            delete domain;
        }
    }
    
    return eIcicleError::SUCCESS;
}

template eIcicleError release_domain_cuda_impl<Fr>();

} // namespace ntt

// =============================================================================
// C++ Mangled Symbol Exports (matching ICICLE)
// =============================================================================

namespace ntt {

// These match the mangled symbols from the library
template<typename F>
eIcicleError ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    F* output
) {
    return ntt_cuda_impl<F>(input, size, direction, config, output);
}

template<typename F>
eIcicleError coset_ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const F& coset_gen,
    const NTTConfig& config,
    F* output
) {
    return coset_ntt_cuda_impl<F>(input, size, direction, coset_gen, config, output);
}

template<typename F>
eIcicleError init_domain_cuda(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
) {
    return init_domain_cuda_impl<F>(root_of_unity, config);
}

template<typename F>
eIcicleError release_domain_cuda() {
    return release_domain_cuda_impl<F>();
}

// Explicit instantiations for symbol export
template eIcicleError ntt_cuda<Fr>(const Fr*, int, NTTDir, const NTTConfig&, Fr*);
template eIcicleError coset_ntt_cuda<Fr>(const Fr*, int, NTTDir, const Fr&, const NTTConfig&, Fr*);
template eIcicleError init_domain_cuda<Fr>(const Fr&, const NTTInitDomainConfig&);
template eIcicleError release_domain_cuda<Fr>();

} // namespace ntt
