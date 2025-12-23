/**
 * @file curve_backend.cu
 * @brief Curve Library Main Entry Points and C API Exports
 * 
 * This file provides the main C API for curve operations exposed to external callers.
 * 
 * ARCHITECTURE:
 * =============
 * CUDA static libraries require kernels to be in the same compilation unit as callers.
 * This file defines its own G1 Montgomery conversion kernels because:
 * 1. The C API functions here call them directly
 * 2. Cannot call kernels from montgomery.cu (different compilation unit)
 * 
 * Kernel duplication between files is INTENTIONAL and REQUIRED for CUDA linking.
 * 
 * This file handles:
 * - MSM entry points (G1 and G2)
 * - G1 Montgomery conversion API (for external use)
 * 
 * Other curve operations are in their respective files:
 * - montgomery.cu: Field-level Montgomery conversions
 * - point_ops.cu: Batch point arithmetic
 * - msm.cu: MSM kernel instantiation
 * - ecntt.cu: EC-NTT operations
 */

#include "field.cuh"
#include "point.cuh"
#include "msm_fwd.cuh"  // Use forward declarations for templated MSM
#include "icicle_types.cuh"

// MSM declarations - defined in msm.cu
namespace msm {
template<typename S, typename A, typename P>
cudaError_t msm_cuda(
    const S* scalars,
    const A* bases,
    int msm_size,
    const icicle::MSMConfig& config,
    P* result
);
}

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// Montgomery Conversion for G1
// =============================================================================

namespace montgomery {

/**
 * @brief Convert G1 projective points to Montgomery form
 */
__global__ void g1_to_montgomery_kernel(
    G1Projective* output,
    const G1Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective p = input[idx];
    field_to_montgomery(p.X, p.X);
    field_to_montgomery(p.Y, p.Y);
    field_to_montgomery(p.Z, p.Z);
    output[idx] = p;
}

/**
 * @brief Convert G1 projective points from Montgomery form
 */
__global__ void g1_from_montgomery_kernel(
    G1Projective* output,
    const G1Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective p = input[idx];
    field_from_montgomery(p.X, p.X);
    field_from_montgomery(p.Y, p.Y);
    field_from_montgomery(p.Z, p.Z);
    output[idx] = p;
}

} // namespace montgomery

// =============================================================================
// C Exports
// =============================================================================

extern "C" {

/**
 * @brief G1 MSM entry point
 */
eIcicleError bls12_381_g1_msm_cuda(
    const Fr* scalars,
    const G1Affine* bases,
    int msm_size,
    const MSMConfig* config,
    G1Projective* result
) {
    cudaError_t err = msm::msm_cuda<Fr, G1Affine, G1Projective>(
        scalars, bases, msm_size, *config, result
    );
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

/**
 * @brief G2 MSM entry point
 * 
 * Now uses the same templated MSM implementation as G1, ensuring:
 * - Constant-time Sort-Reduce algorithm (security)
 * - Consistent behavior across G1 and G2
 * - Proper Fq2 field arithmetic via point_add, point_double overloads
 */
eIcicleError bls12_381_g2_msm_cuda(
    const Fr* scalars,
    const G2Affine* bases,
    int msm_size,
    const MSMConfig* config,
    G2Projective* result
) {
    cudaError_t err = msm::msm_cuda<Fr, G2Affine, G2Projective>(
        scalars, bases, msm_size, *config, result
    );
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

/**
 * @brief G1 Montgomery conversion entry point
 */
eIcicleError bls12_381_g1_to_montgomery(
    const G1Projective* input,
    int size,
    const VecOpsConfig* config,
    G1Projective* output
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    
    const G1Projective* d_input = input;
    G1Projective* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_input) {
        CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(G1Projective)));
        CUDA_CHECK(cudaMemcpyAsync((void*)d_input, input, size * sizeof(G1Projective), cudaMemcpyHostToDevice, stream));
    }
    if (need_alloc_output) {
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(G1Projective)));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    montgomery::g1_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_input, size
    );
    CUDA_CHECK(cudaGetLastError());
    
    if (need_alloc_output) {
        CUDA_CHECK(cudaMemcpyAsync(output, d_output, size * sizeof(G1Projective), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaFree(d_output));
    }
    if (need_alloc_input) {
        CUDA_CHECK(cudaFree((void*)d_input));
    }
    
    if (!config->is_async) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief G1 from Montgomery conversion entry point
 */
eIcicleError bls12_381_g1_from_montgomery(
    const G1Projective* input,
    int size,
    const VecOpsConfig* config,
    G1Projective* output
) {
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    cudaError_t err;
    
    const G1Projective* d_input = input;
    G1Projective* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_input) {
        err = cudaMalloc((void**)&d_input, size * sizeof(G1Projective));
        if (err != cudaSuccess) return eIcicleError::ALLOCATION_FAILED;
        
        // Use async copy with stream
        err = cudaMemcpyAsync((void*)d_input, input, size * sizeof(G1Projective), 
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    if (need_alloc_output) {
        err = cudaMalloc(&d_output, size * sizeof(G1Projective));
        if (err != cudaSuccess) {
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::ALLOCATION_FAILED;
        }
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    montgomery::g1_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_input, size
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (need_alloc_output) cudaFree(d_output);
        if (need_alloc_input) cudaFree((void*)d_input);
        return eIcicleError::UNKNOWN_ERROR;
    }
    
    if (need_alloc_output) {
        // Use async copy with stream
        err = cudaMemcpyAsync(output, d_output, size * sizeof(G1Projective), 
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            cudaFree(d_output);
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    
    // Synchronize if not async mode
    if (!config->is_async) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            if (need_alloc_output) cudaFree(d_output);
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::UNKNOWN_ERROR;
        }
    }
    
    // Cleanup allocations (must sync before freeing in async mode)
    if (config->is_async && (need_alloc_input || need_alloc_output)) {
        cudaStreamSynchronize(stream);
    }
    
    if (need_alloc_output) {
        cudaFree(d_output);
    }
    if (need_alloc_input) {
        cudaFree((void*)d_input);
    }
    
    return eIcicleError::SUCCESS;
}

} // extern "C"
