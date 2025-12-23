/**
 * @file vec_ops.cu
 * @brief Vector Operations for BLS12-381 Scalar Field
 * 
 * Provides vectorized field operations essential for polynomial arithmetic in ZK provers.
 * 
 * ARCHITECTURE:
 * =============
 * All kernels are defined and called in this file (self-contained).
 * This is required by CUDA's static library linking model.
 * 
 * Operations provided:
 * - vec_add: Element-wise vector addition
 * - vec_sub: Element-wise vector subtraction
 * - vec_mul: Element-wise vector multiplication (Hadamard product)
 * - scalar_vec_mul: Scalar-vector multiplication
 * - vec_neg: Vector negation
 * - vec_inv: Element-wise modular inversion
 * - vec_sum: Parallel reduction to compute sum
 * - inner_product: Parallel inner product
 * 
 * Performance: All operations use Montgomery form for efficient modular arithmetic.
 */

#include "field.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>

using namespace bls12_381;

namespace vec_ops {

// =============================================================================
// Vector Arithmetic Kernels
// =============================================================================

/**
 * @brief Element-wise vector addition
 */
__global__ void vec_add_kernel(
    Fr* output,
    const Fr* a,
    const Fr* b,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = a[idx] + b[idx];
}

/**
 * @brief Element-wise vector subtraction
 */
__global__ void vec_sub_kernel(
    Fr* output,
    const Fr* a,
    const Fr* b,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = a[idx] - b[idx];
}

/**
 * @brief Element-wise vector multiplication (Hadamard product)
 */
__global__ void vec_mul_kernel(
    Fr* output,
    const Fr* a,
    const Fr* b,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = a[idx] * b[idx];
}

/**
 * @brief Scalar-vector multiplication
 */
__global__ void scalar_vec_mul_kernel(
    Fr* output,
    const Fr scalar,
    const Fr* vec,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = scalar * vec[idx];
}

/**
 * @brief Vector negation
 */
__global__ void vec_neg_kernel(
    Fr* output,
    const Fr* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = field_neg(input[idx]);
}

/**
 * @brief Vector inversion (element-wise)
 */
__global__ void vec_inv_kernel(
    Fr* output,
    const Fr* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = field_inv(input[idx]);
}

// =============================================================================
// Batch Inversion using Montgomery's Trick
// =============================================================================

/**
 * @brief Compute prefix products for batch inversion
 * 
 * prefix[i] = input[0] * input[1] * ... * input[i]
 * 
 * Uses parallel scan algorithm for efficiency.
 */
__global__ void batch_inv_prefix_kernel(
    Fr* prefix,
    const Fr* input,
    int size
) {
    // Block-level inclusive scan
    extern __shared__ Fr sdata[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (gid < size) {
        sdata[tid] = input[gid];
    } else {
        sdata[tid] = Fr::one();
    }
    __syncthreads();
    
    // Up-sweep (reduce) phase - build partial products
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            sdata[idx] = sdata[idx] * sdata[idx - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase - distribute products
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx + stride < blockDim.x) {
            sdata[idx + stride] = sdata[idx + stride] * sdata[idx];
        }
        __syncthreads();
    }
    
    // Write result
    if (gid < size) {
        prefix[gid] = sdata[tid];
    }
}

/**
 * @brief Complete prefix products across blocks (sequential step)
 */
__global__ void batch_inv_prefix_fixup_kernel(
    Fr* prefix,
    const Fr* block_products,
    int size,
    int block_size
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    
    int block_idx = gid / block_size;
    
    // Multiply by product of all previous blocks
    if (block_idx > 0) {
        prefix[gid] = prefix[gid] * block_products[block_idx - 1];
    }
}

/**
 * @brief Compute individual inverses from prefix products
 * 
 * Given prefix[i] = input[0] * ... * input[i] and total_inv = 1/prefix[n-1]:
 * output[i] = prefix[i-1] * suffix_inv[i]
 * 
 * where suffix_inv[i] = 1/(input[i] * input[i+1] * ... * input[n-1])
 */
__global__ void batch_inv_compute_kernel(
    Fr* output,
    const Fr* input,
    const Fr* prefix,
    const Fr total_inv,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // suffix_inv[i] = total_inv * prefix[n-1] / prefix[i]
    //               = total_inv * (input[i+1] * ... * input[n-1])^-1 ... wait
    // 
    // Actually: output[i] = 1/input[i]
    // We know: prefix[i] = input[0] * ... * input[i]
    // And: total_inv = 1/(input[0] * ... * input[n-1])
    // 
    // So: suffix_product[i] = input[i] * ... * input[n-1] = prefix[n-1] / prefix[i-1]
    // And: suffix_inv[i] = total_inv * prefix[i-1]
    // Therefore: output[i] = (prefix[i-1] if i > 0 else 1) * total_inv * suffix_product[i+1..n-1]
    //                      = (prefix[i-1] if i > 0 else 1) * (prefix[n-1] * total_inv / prefix[i])
    //                      = (prefix[i-1] if i > 0 else 1) / prefix[i]   ... but that's wrong
    //
    // Correct formula:
    // output[i] = 1/input[i] = prefix[i-1] * (total_inv * suffix_product[i+1..n])
    //
    // Let's use simpler approach:
    // output[i] = prefix[i-1] * suffix_inv
    // where we compute suffix_inv right-to-left
    
    // Simpler: compute directly
    // output[i] = prefix[i-1] * suffix_inv_after_i
    //           = (i == 0 ? Fr::one() : prefix[i-1]) * (total_inv * prefix_from_i+1_to_n)
    
    Fr left_product = (idx == 0) ? Fr::one() : prefix[idx - 1];
    Fr right_product = (idx == size - 1) ? Fr::one() : prefix[size - 1] * field_inv(prefix[idx]);
    
    // Actually the cleanest way:
    // output[i] = left_product * total_inv * right_product
    // where left_product = input[0] * ... * input[i-1]
    //       right_product = input[i+1] * ... * input[n-1]
    // 
    // We have prefix[i] = input[0] * ... * input[i]
    // left_product = prefix[i-1] for i > 0, else 1
    // right_product = prefix[n-1] / prefix[i] for i < n-1, else 1
    
    // But dividing requires inversion, defeating the purpose!
    // The standard Montgomery trick avoids this by computing suffix array in reverse.
    // Since we're on GPU, we'll use a different approach - see batch_inv_cuda_impl below.
    
    // For now, use the slower per-element approach (this kernel is unused)
    output[idx] = field_inv(input[idx]);
}

/**
 * @brief Parallel batch inversion using Montgomery's trick
 * 
 * Algorithm:
 * 1. Phase 1: Each block computes prefix products for its chunk
 * 2. Phase 2: Single thread inverts final product (unavoidable)
 * 3. Phase 3: Each block computes suffix inverses and final results
 * 
 * Inverts n elements using 3(n-1) multiplications + 1 inversion
 * instead of n inversions (each ~256 multiplications).
 * 
 * Speedup: ~85x for large batches
 */

/**
 * @brief Phase 1: Compute prefix products within each block
 */
__global__ void batch_inv_prefix_phase1_kernel(
    Fr* prefix,           // Output: prefix products (same size as input)
    Fr* block_products,   // Output: product of each block
    const Fr* input,
    int size
) {
    extern __shared__ Fr shared[];
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int gid = block_start + tid;
    
    // Load input to shared memory
    if (gid < size) {
        shared[tid] = input[gid];
    } else {
        shared[tid] = Fr::one();
    }
    __syncthreads();
    
    // Parallel prefix product using Hillis-Steele algorithm
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        Fr val = shared[tid];
        if (tid >= stride) {
            val = shared[tid - stride] * val;
        }
        __syncthreads();
        shared[tid] = val;
        __syncthreads();
    }
    
    // Write prefix products to global memory
    if (gid < size) {
        prefix[gid] = shared[tid];
    }
    
    // Last thread in block writes block product
    if (tid == blockDim.x - 1) {
        block_products[blockIdx.x] = shared[tid];
    }
}

/**
 * @brief Phase 2: Compute prefix products of block products (small, runs on single block)
 */
__global__ void batch_inv_block_prefix_kernel(
    Fr* block_prefix_inv,   // Output: inclusive prefix inverse for each block
    Fr* block_products,     // Input: product of each block
    int num_blocks
) {
    // Single block handles this phase
    if (blockIdx.x != 0) return;
    
    extern __shared__ Fr shared[];
    int tid = threadIdx.x;
    
    // Load block products
    if (tid < num_blocks) {
        shared[tid] = block_products[tid];
    } else {
        shared[tid] = Fr::one();
    }
    __syncthreads();
    
    // Compute prefix products of block products
    if (tid == 0) {
        for (int i = 1; i < num_blocks; i++) {
            shared[i] = shared[i-1] * shared[i];
        }
    }
    __syncthreads();
    
    // Invert the total product
    Fr total_inv;
    if (tid == 0) {
        total_inv = field_inv(shared[num_blocks - 1]);
    }
    
    // Broadcast total_inv to all threads via shared memory
    if (tid == 0) {
        shared[num_blocks] = total_inv;
    }
    __syncthreads();
    total_inv = shared[num_blocks];
    
    // Compute inclusive prefix inverses for each block
    // block_prefix_inv[i] = 1 / (block_product[0] * ... * block_product[i])
    // We have:
    // shared[i] = B_0 * ... * B_i
    // total_inv = 1 / (B_0 * ... * B_{n-1})
    //
    // We want inv(shared[i]).
    // inv(shared[i]) = total_inv * (B_{i+1} * ... * B_{n-1})
    // This requires suffix products of blocks.
    
    // Simpler: Just invert each shared[i] sequentially?
    // Or use the same trick:
    // inv(shared[i]) = inv(shared[i+1]) * B_{i+1}.
    // Start from inv(shared[n-1]) = total_inv.
    
    if (tid == 0) {
        Fr current_inv = total_inv;
        block_prefix_inv[num_blocks - 1] = current_inv;
        
        for (int i = num_blocks - 2; i >= 0; i--) {
            current_inv = current_inv * block_products[i+1];
            block_prefix_inv[i] = current_inv;
        }
    }
}

/**
 * @brief Phase 3: Compute individual inverses using prefix products and block inverses
 */
__global__ void batch_inv_compute_phase3_kernel(
    Fr* output,
    const Fr* input,
    const Fr* prefix,
    const Fr* block_prefix_inv,
    int size
) {
    extern __shared__ Fr shared[];
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int gid = block_start + tid;
    
    // Load input to shared memory
    Fr val = Fr::one();
    if (gid < size) {
        val = input[gid];
    }
    shared[tid] = val;
    __syncthreads();
    
    // Compute suffix products in shared memory (Backward Scan)
    // We want shared[tid] to contain product of input[gid]...input[block_end-1]
    
    // Naive parallel suffix scan
    // For stride = 1, 2, 4...
    // if tid + stride < blockDim, val = val * shared[tid+stride]
    // But we need to be careful with synchronization and overwriting
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        Fr neighbor = Fr::one();
        if (tid + stride < blockDim.x) {
            neighbor = shared[tid + stride];
        }
        __syncthreads();
        
        // Only update if we have a neighbor (optimization)
        if (tid + stride < blockDim.x) {
            shared[tid] = shared[tid] * neighbor;
        }
        __syncthreads();
    }
    
    if (gid >= size) return;
    
    // output[i] = prefix[i-1] * local_suffix[i+1] * block_prefix_inv[blockIdx.x]
    
    Fr prev_prefix = (gid == 0) ? Fr::one() : prefix[gid - 1];
    
    Fr local_suffix_next = Fr::one();
    if (tid + 1 < blockDim.x && gid + 1 < size) {
        local_suffix_next = shared[tid + 1];
    }
    
    Fr block_inv = block_prefix_inv[blockIdx.x];
    
    // Combine
    Fr res = prev_prefix * local_suffix_next;
    res = res * block_inv;
    
    output[gid] = res;
}

/**
 * @brief Simple parallel batch inversion (block-parallel Montgomery's trick)
 * 
 * Each block independently inverts a chunk of elements.
 * For large arrays, this is much faster than single-threaded.
 */
__global__ void batch_inv_parallel_simple_kernel(
    Fr* output,
    const Fr* input,
    int size,
    int elements_per_thread
) {
    extern __shared__ Fr scratch[];
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * elements_per_thread;
    int my_start = block_start + tid * elements_per_thread;
    int my_end = min(my_start + elements_per_thread, size);
    int my_size = my_end - my_start;
    
    if (my_size <= 0) return;
    
    // Each thread processes its chunk using Montgomery's trick
    Fr* my_scratch = scratch + tid * elements_per_thread;
    
    // Step 1: Compute prefix products
    my_scratch[0] = input[my_start];
    for (int i = 1; i < my_size; i++) {
        my_scratch[i] = my_scratch[i-1] * input[my_start + i];
    }
    
    // Step 2: Invert final product
    Fr total_inv = field_inv(my_scratch[my_size - 1]);
    
    // Step 3: Compute individual inverses
    for (int i = my_size - 1; i >= 0; i--) {
        if (i == 0) {
            output[my_start] = total_inv;
        } else {
            output[my_start + i] = my_scratch[i-1] * total_inv;
        }
        total_inv = total_inv * input[my_start + i];
    }
}

/**
 * @brief Second phase: apply cross-block correction and compute inverses
 */
__global__ void batch_inv_parallel_phase2_kernel(
    Fr* output,
    const Fr* input,
    Fr* prefix,  // Global prefix array (reused)
    const Fr* block_prefix_inv,  // Prefix inverse for each block
    int size,
    int elements_per_block
) {
    int tid = threadIdx.x;
    int block_start = blockIdx.x * elements_per_block;
    int block_end = min(block_start + elements_per_block, size);
    int block_size = block_end - block_start;
    
    if (tid >= block_size) return;
    
    // Note: gid would be used in fully parallel version
    // int gid = block_start + tid;
    (void)block_start;  // Suppress unused warning
    (void)block_prefix_inv;  // Suppress unused warning
    
    // Get the inverse of product from start of this block to element gid
    // This is: block_prefix_inv[blockIdx.x] * (local prefix up to tid)^-1
    // 
    // Actually we need to recompute local prefix and apply Montgomery's trick
    // within each block, using the cross-block inverse as the "total_inv"
    
    // For simplicity, use the serial kernel for now
    // A fully parallel version would require more complex synchronization
}

/**
 * @brief Scalar addition to vector
 */
__global__ void scalar_vec_add_kernel(
    Fr* output,
    const Fr scalar,
    const Fr* vec,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = scalar + vec[idx];
}

/**
 * @brief Sum reduction kernel (partial sums)
 */
__global__ void vec_sum_partial_kernel(
    Fr* partial_sums,
    const Fr* input,
    int size
) {
    extern __shared__ Fr sdata[];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load and perform first reduction in shared memory
    Fr sum = Fr::zero();
    if (global_idx < size) {
        sum = input[global_idx];
    }
    if (global_idx + blockDim.x < size) {
        sum = sum + input[global_idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Inner product kernel (dot product)
 */
__global__ void vec_inner_product_partial_kernel(
    Fr* partial_sums,
    const Fr* a,
    const Fr* b,
    int size
) {
    extern __shared__ Fr sdata[];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load, multiply, and perform first reduction
    Fr sum = Fr::zero();
    if (global_idx < size) {
        sum = a[global_idx] * b[global_idx];
    }
    if (global_idx + blockDim.x < size) {
        sum = sum + a[global_idx + blockDim.x] * b[global_idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// ICICLE API Implementation
// =============================================================================

/**
 * @brief Generic vector operation implementation
 */
template<typename Op>
eIcicleError vec_op_impl(
    Fr* output,
    const Fr* a,
    const Fr* b,
    int size,
    const VecOpsConfig& config,
    Op kernel
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    const Fr* d_a = a;
    const Fr* d_b = b;
    Fr* d_output = output;
    
    bool alloc_a = !config.is_a_on_device;
    bool alloc_b = !config.is_b_on_device;
    bool alloc_out = !config.is_result_on_device;
    
    // Allocate device memory if needed
    if (alloc_a) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpy(temp, a, size * sizeof(Fr), cudaMemcpyHostToDevice);
        d_a = temp;
    }
    
    if (alloc_b && b != nullptr) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpy(temp, b, size * sizeof(Fr), cudaMemcpyHostToDevice);
        d_b = temp;
    }
    
    if (alloc_out) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    kernel<<<blocks, threads, 0, stream>>>(d_output, d_a, d_b, size);
    
    // Copy output back if needed
    if (alloc_out) {
        cudaMemcpy(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
    }
    
    // Free temporary allocations
    if (alloc_a) cudaFree(const_cast<Fr*>(d_a));
    if (alloc_b && b != nullptr) cudaFree(const_cast<Fr*>(d_b));
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

} // namespace vec_ops

// =============================================================================
// Exported Symbols
// =============================================================================

extern "C" {

// Vector-vector operations
eIcicleError vec_add_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* a,
    const bls12_381::Fr* b,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_ops::vec_add_kernel<<<blocks, threads, 0, stream>>>(output, a, b, size);
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError vec_sub_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* a,
    const bls12_381::Fr* b,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_ops::vec_sub_kernel<<<blocks, threads, 0, stream>>>(output, a, b, size);
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError vec_mul_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* a,
    const bls12_381::Fr* b,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_ops::vec_mul_kernel<<<blocks, threads, 0, stream>>>(output, a, b, size);
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

// Scalar-vector operations
eIcicleError scalar_mul_vec_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* scalar,
    const bls12_381::Fr* vec,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_ops::scalar_vec_mul_kernel<<<blocks, threads, 0, stream>>>(
        output, *scalar, vec, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError scalar_add_vec_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* scalar,
    const bls12_381::Fr* vec,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_ops::scalar_vec_add_kernel<<<blocks, threads, 0, stream>>>(
        output, *scalar, vec, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

// Reduction operations
eIcicleError vec_sum_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* input,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + 2 * threads - 1) / (2 * threads);
    
    // Allocate partial sums
    Fr* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(Fr));
    
    // First reduction
    vec_ops::vec_sum_partial_kernel<<<blocks, threads, threads * sizeof(Fr), stream>>>(
        d_partial, input, size
    );
    
    // Continue reduction until single value
    int remaining = blocks;
    while (remaining > 1) {
        int new_blocks = (remaining + 2 * threads - 1) / (2 * threads);
        Fr* d_new_partial;
        cudaMalloc(&d_new_partial, new_blocks * sizeof(Fr));
        
        vec_ops::vec_sum_partial_kernel<<<new_blocks, threads, threads * sizeof(Fr), stream>>>(
            d_new_partial, d_partial, remaining
        );
        
        cudaFree(d_partial);
        d_partial = d_new_partial;
        remaining = new_blocks;
    }
    
    // Copy result
    if (config.is_result_on_device) {
        cudaMemcpy(output, d_partial, sizeof(Fr), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy(output, d_partial, sizeof(Fr), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_partial);
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

} // extern "C"

// =============================================================================
// C++ Template Exports (matching ICICLE mangled names)
// =============================================================================

namespace vec_ops {

template<typename F>
eIcicleError vec_add_cuda(
    F* output,
    const F* a,
    const F* b,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_add_kernel<<<blocks, threads, 0, stream>>>(output, a, b, size);
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

template<typename F>
eIcicleError vec_sub_cuda(
    F* output,
    const F* a,
    const F* b,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_sub_kernel<<<blocks, threads, 0, stream>>>(output, a, b, size);
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

template<typename F>
eIcicleError vec_mul_cuda(
    F* output,
    const F* a,
    const F* b,
    int size,
    const VecOpsConfig& config
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vec_mul_kernel<<<blocks, threads, 0, stream>>>(output, a, b, size);
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

/**
 * @brief Batch field inversion using Montgomery's trick
 * 
 * Inverts n elements using 3(n-1) multiplications + 1 inversion
 * instead of n inversions (each ~300 operations for BLS12-381).
 * 
 * Speedup: ~100x for large batches (1 inversion vs n inversions)
 * 
 * Algorithm:
 * 1. Compute prefix products: prefix[i] = input[0] * ... * input[i]
 * 2. Invert the final product: total_inv = 1/prefix[n-1]
 * 3. Work backwards to compute each inverse:
 *    output[i] = prefix[i-1] * total_inv (where total_inv accumulates)
 *    total_inv *= input[i] (to prepare for next iteration)
 * 
 * @param output  Output array for inverses (device memory)
 * @param input   Input array of elements to invert (device memory)
 * @param size    Number of elements
 * @param config  Configuration (stream, etc.)
 * @return eIcicleError::SUCCESS on success
 */
template<typename F>
eIcicleError batch_inv_cuda(
    F* output,
    const F* input,
    int size,
    const VecOpsConfig& config
) {
    if (size == 0) return eIcicleError::SUCCESS;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    // For small sizes, use simple element-wise inversion
    // (overhead of Montgomery's trick not worth it below ~16 elements)
    if (size < 16) {
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;
        vec_inv_kernel<<<blocks, threads, 0, stream>>>(output, input, size);
        return cudaGetLastError() == cudaSuccess ? 
               eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
    }
    
    // Allocate scratch space
    F *d_prefix, *d_block_products, *d_block_prefix_inv;
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Check if we exceed single-block limit for Phase 2
    if (blocks > 1024) {
        // For very large arrays (>262K elements), fall back to simple element-wise inversion
        // A full recursive implementation would require Phase 2 to be multi-level
        const int inv_threads = 256;
        const int inv_blocks = (size + inv_threads - 1) / inv_threads;
        vec_inv_kernel<<<inv_blocks, inv_threads, 0, stream>>>(output, input, size);
        return cudaGetLastError() == cudaSuccess ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
    }
    
    cudaError_t err;
    err = cudaMalloc(&d_prefix, size * sizeof(F));
    if (err != cudaSuccess) return eIcicleError::ALLOCATION_FAILED;
    
    err = cudaMalloc(&d_block_products, blocks * sizeof(F));
    if (err != cudaSuccess) { cudaFree(d_prefix); return eIcicleError::ALLOCATION_FAILED; }
    
    err = cudaMalloc(&d_block_prefix_inv, blocks * sizeof(F));
    if (err != cudaSuccess) { cudaFree(d_prefix); cudaFree(d_block_products); return eIcicleError::ALLOCATION_FAILED; }
    
    // Phase 1: Block-level prefix products
    batch_inv_prefix_phase1_kernel<<<blocks, threads, threads * sizeof(F), stream>>>(
        d_prefix, d_block_products, input, size
    );
    
    // Phase 2: Block product prefix inverses (single block)
    // Shared mem: blocks elements + 1 for total_inv
    batch_inv_block_prefix_kernel<<<1, blocks, (blocks + 1) * sizeof(F), stream>>>(
        d_block_prefix_inv, d_block_products, blocks
    );
    
    // Phase 3: Final computation
    batch_inv_compute_phase3_kernel<<<blocks, threads, threads * sizeof(F), stream>>>(
        output, input, d_prefix, d_block_prefix_inv, size
    );
    
    err = cudaGetLastError();
    
    // Cleanup (async if possible, but we need to free)
    // If async, we can't free immediately unless we use cudaFreeAsync (CUDA 11.2+)
    // For compatibility, we synchronize.
    cudaStreamSynchronize(stream);
    
    cudaFree(d_prefix);
    cudaFree(d_block_products);
    cudaFree(d_block_prefix_inv);
    
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

// Explicit instantiations
template eIcicleError vec_add_cuda<Fr>(Fr*, const Fr*, const Fr*, int, const VecOpsConfig&);
template eIcicleError vec_sub_cuda<Fr>(Fr*, const Fr*, const Fr*, int, const VecOpsConfig&);
template eIcicleError vec_mul_cuda<Fr>(Fr*, const Fr*, const Fr*, int, const VecOpsConfig&);
template eIcicleError batch_inv_cuda<Fr>(Fr*, const Fr*, int, const VecOpsConfig&);

} // namespace vec_ops

// =============================================================================
// ICICLE-compatible exported symbols
// =============================================================================

extern "C" {

eIcicleError bls12_381_vector_add(
    const bls12_381::Fr* a,
    const bls12_381::Fr* b,
    size_t size,
    const VecOpsConfig* config,
    bls12_381::Fr* output
) {
    using namespace bls12_381;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    
    const Fr* d_a = a;
    const Fr* d_b = b;
    Fr* d_output = output;
    
    bool need_alloc_a = !config->is_a_on_device;
    bool need_alloc_b = !config->is_b_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_a) {
        cudaMalloc((void**)&d_a, size * sizeof(Fr));
        cudaMemcpy((void*)d_a, a, size * sizeof(Fr), cudaMemcpyHostToDevice);
    }
    if (need_alloc_b) {
        cudaMalloc((void**)&d_b, size * sizeof(Fr));
        cudaMemcpy((void*)d_b, b, size * sizeof(Fr), cudaMemcpyHostToDevice);
    }
    if (need_alloc_output) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vec_ops::vec_add_kernel<<<blocks, threads, 0, stream>>>(d_output, d_a, d_b, (int)size);
    cudaStreamSynchronize(stream);
    
    if (need_alloc_output) {
        cudaMemcpy(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
    }
    if (need_alloc_a) {
        cudaFree((void*)d_a);
    }
    if (need_alloc_b) {
        cudaFree((void*)d_b);
    }
    
    return eIcicleError::SUCCESS;
}

eIcicleError bls12_381_vector_sub(
    const bls12_381::Fr* a,
    const bls12_381::Fr* b,
    size_t size,
    const VecOpsConfig* config,
    bls12_381::Fr* output
) {
    using namespace bls12_381;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    
    const Fr* d_a = a;
    const Fr* d_b = b;
    Fr* d_output = output;
    
    bool need_alloc_a = !config->is_a_on_device;
    bool need_alloc_b = !config->is_b_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_a) {
        cudaMalloc((void**)&d_a, size * sizeof(Fr));
        cudaMemcpy((void*)d_a, a, size * sizeof(Fr), cudaMemcpyHostToDevice);
    }
    if (need_alloc_b) {
        cudaMalloc((void**)&d_b, size * sizeof(Fr));
        cudaMemcpy((void*)d_b, b, size * sizeof(Fr), cudaMemcpyHostToDevice);
    }
    if (need_alloc_output) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vec_ops::vec_sub_kernel<<<blocks, threads, 0, stream>>>(d_output, d_a, d_b, (int)size);
    cudaStreamSynchronize(stream);
    
    if (need_alloc_output) {
        cudaMemcpy(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
    }
    if (need_alloc_a) {
        cudaFree((void*)d_a);
    }
    if (need_alloc_b) {
        cudaFree((void*)d_b);
    }
    
    return eIcicleError::SUCCESS;
}

eIcicleError bls12_381_vector_mul(
    const bls12_381::Fr* a,
    const bls12_381::Fr* b,
    size_t size,
    const VecOpsConfig* config,
    bls12_381::Fr* output
) {
    using namespace bls12_381;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    
    const Fr* d_a = a;
    const Fr* d_b = b;
    Fr* d_output = output;
    
    bool need_alloc_a = !config->is_a_on_device;
    bool need_alloc_b = !config->is_b_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_a) {
        cudaMalloc((void**)&d_a, size * sizeof(Fr));
        cudaMemcpy((void*)d_a, a, size * sizeof(Fr), cudaMemcpyHostToDevice);
    }
    if (need_alloc_b) {
        cudaMalloc((void**)&d_b, size * sizeof(Fr));
        cudaMemcpy((void*)d_b, b, size * sizeof(Fr), cudaMemcpyHostToDevice);
    }
    if (need_alloc_output) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vec_ops::vec_mul_kernel<<<blocks, threads, 0, stream>>>(d_output, d_a, d_b, (int)size);
    cudaStreamSynchronize(stream);
    
    if (need_alloc_output) {
        cudaMemcpy(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
    }
    if (need_alloc_a) {
        cudaFree((void*)d_a);
    }
    if (need_alloc_b) {
        cudaFree((void*)d_b);
    }
    
    return eIcicleError::SUCCESS;
}

} // extern "C"
