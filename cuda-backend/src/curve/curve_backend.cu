/**
 * @file curve_backend.cu
 * @brief Curve library exports for Icicle compatibility
 * 
 * Full implementation of G1 and G2 operations.
 */

#include "field.cuh"
#include "point.cuh"
#include "msm_g2.cuh"
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
 */
eIcicleError bls12_381_g2_msm_cuda(
    const Fr* scalars,
    const G2Affine* bases,
    int msm_size,
    const MSMConfig* config,
    G2Projective* result
) {
    cudaError_t err = msm_g2::g2_msm_cuda(scalars, bases, msm_size, *config, result);
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
    
    const G1Projective* d_input = input;
    G1Projective* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_input) {
        cudaMalloc((void**)&d_input, size * sizeof(G1Projective));
        cudaMemcpy((void*)d_input, input, size * sizeof(G1Projective), cudaMemcpyHostToDevice);
    }
    if (need_alloc_output) {
        cudaMalloc(&d_output, size * sizeof(G1Projective));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    montgomery::g1_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_input, size
    );
    
    if (need_alloc_output) {
        cudaMemcpy(output, d_output, size * sizeof(G1Projective), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
    }
    if (need_alloc_input) {
        cudaFree((void*)d_input);
    }
    
    return eIcicleError::SUCCESS;
}

} // extern "C"
