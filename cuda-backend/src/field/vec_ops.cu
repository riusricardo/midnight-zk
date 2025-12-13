/**
 * @file vec_ops.cu
 * @brief Vector operations for BLS12-381 scalar field
 * 
 * This file provides vectorized field operations that are essential
 * for polynomial arithmetic in the prover.
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

// Explicit instantiations
template eIcicleError vec_add_cuda<Fr>(Fr*, const Fr*, const Fr*, int, const VecOpsConfig&);
template eIcicleError vec_sub_cuda<Fr>(Fr*, const Fr*, const Fr*, int, const VecOpsConfig&);
template eIcicleError vec_mul_cuda<Fr>(Fr*, const Fr*, const Fr*, int, const VecOpsConfig&);

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
