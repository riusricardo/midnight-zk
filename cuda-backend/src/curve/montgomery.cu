/**
 * @file montgomery.cu
 * @brief Montgomery Form Conversion Utilities
 * 
 * Provides kernels and API functions for converting between standard and
 * Montgomery representation for field elements and curve points.
 * 
 * ARCHITECTURE:
 * =============
 * This file is self-contained with its own kernels because CUDA static libraries
 * require kernels to be defined in the same compilation unit that calls them.
 * 
 * Note: curve_backend.cu has similar G1 Projective Montgomery kernels - this is
 * intentional duplication required by CUDA's linking model, not a bug.
 * 
 * Kernels defined here:
 * - fr_to_montgomery_kernel / fr_from_montgomery_kernel (scalar field)
 * - fq_to_montgomery_kernel / fq_from_montgomery_kernel (base field)
 * - g1_affine_to/from_montgomery_kernel (G1 affine points)
 * - g1_projective_to/from_montgomery_kernel (G1 projective points)
 * 
 * Montgomery form: a_mont = a * R mod p (where R = 2^256 for Fr, 2^384 for Fq)
 */

#include "field.cuh"
#include "point.cuh"
#include "icicle_types.cuh"

namespace montgomery {

using namespace bls12_381;

// =============================================================================
// Field Montgomery Conversion Kernels
// =============================================================================

/**
 * @brief Convert Fr elements from standard to Montgomery form
 * 
 * Montgomery form: a_mont = a * R mod p
 * where R = 2^256 for Fr
 */
__global__ void fr_to_montgomery_kernel(
    Fr* output,
    const Fr* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    Fr val = input[idx];
    
    // Multiply by R^2 and then reduce
    // to_montgomery(a) = a * R^2 * R^(-1) mod p = a * R mod p
    output[idx] = val * Fr::R_SQUARED();
}

/**
 * @brief Convert Fr elements from Montgomery to standard form
 * 
 * Standard form: a = a_mont * R^(-1) mod p
 */
__global__ void fr_from_montgomery_kernel(
    Fr* output,
    const Fr* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    Fr val = input[idx];
    
    // Montgomery reduction: multiply by 1 and reduce
    // from_montgomery(a_mont) = a_mont * 1 * R^(-1) mod p = a mod p
    Fr one;
    for (int i = 0; i < Fr::LIMBS; i++) {
        one.limbs[i] = (i == 0) ? 1 : 0;
    }
    
    output[idx] = val * one;
}

/**
 * @brief Convert Fq elements to Montgomery form
 */
__global__ void fq_to_montgomery_kernel(
    Fq* output,
    const Fq* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = input[idx] * Fq::R_SQUARED();
}

/**
 * @brief Convert Fq elements from Montgomery form
 */
__global__ void fq_from_montgomery_kernel(
    Fq* output,
    const Fq* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    Fq val = input[idx];
    
    Fq one;
    for (int i = 0; i < Fq::LIMBS; i++) {
        one.limbs[i] = (i == 0) ? 1 : 0;
    }
    
    output[idx] = val * one;
}

// =============================================================================
// Point Montgomery Conversion Kernels
// =============================================================================

/**
 * @brief Convert G1 affine points to Montgomery form
 */
__global__ void g1_affine_to_montgomery_kernel(
    G1Affine* output,
    const G1Affine* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Affine p = input[idx];
    
    // Convert x and y coordinates
    p.x = p.x * Fq::R_SQUARED();
    p.y = p.y * Fq::R_SQUARED();
    
    output[idx] = p;
}

/**
 * @brief Convert G1 affine points from Montgomery form
 */
__global__ void g1_affine_from_montgomery_kernel(
    G1Affine* output,
    const G1Affine* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Affine p = input[idx];
    
    Fq one;
    for (int i = 0; i < Fq::LIMBS; i++) {
        one.limbs[i] = (i == 0) ? 1 : 0;
    }
    
    p.x = p.x * one;
    p.y = p.y * one;
    
    output[idx] = p;
}

/**
 * @brief Convert G1 projective points to Montgomery form
 */
__global__ void g1_projective_to_montgomery_kernel(
    G1Projective* output,
    const G1Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective p = input[idx];
    
    p.X = p.X * Fq::R_SQUARED();
    p.Y = p.Y * Fq::R_SQUARED();
    p.Z = p.Z * Fq::R_SQUARED();
    
    output[idx] = p;
}

/**
 * @brief Convert G1 projective points from Montgomery form
 */
__global__ void g1_projective_from_montgomery_kernel(
    G1Projective* output,
    const G1Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective p = input[idx];
    
    Fq one;
    for (int i = 0; i < Fq::LIMBS; i++) {
        one.limbs[i] = (i == 0) ? 1 : 0;
    }
    
    p.X = p.X * one;
    p.Y = p.Y * one;
    p.Z = p.Z * one;
    
    output[idx] = p;
}

} // namespace montgomery

// =============================================================================
// ICICLE API Exports
// =============================================================================

extern "C" {

// Fr Montgomery conversions
eIcicleError fr_to_montgomery_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::fr_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError fr_from_montgomery_cuda(
    bls12_381::Fr* output,
    const bls12_381::Fr* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::fr_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

// Fq Montgomery conversions
eIcicleError fq_to_montgomery_cuda(
    bls12_381::Fq* output,
    const bls12_381::Fq* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::fq_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError fq_from_montgomery_cuda(
    bls12_381::Fq* output,
    const bls12_381::Fq* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::fq_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

// G1 Affine Montgomery conversions
eIcicleError g1_affine_to_montgomery_cuda(
    bls12_381::G1Affine* output,
    const bls12_381::G1Affine* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::g1_affine_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError g1_affine_from_montgomery_cuda(
    bls12_381::G1Affine* output,
    const bls12_381::G1Affine* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::g1_affine_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

// G1 Projective Montgomery conversions
eIcicleError g1_projective_to_montgomery_cuda(
    bls12_381::G1Projective* output,
    const bls12_381::G1Projective* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::g1_projective_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError g1_projective_from_montgomery_cuda(
    bls12_381::G1Projective* output,
    const bls12_381::G1Projective* input,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    montgomery::g1_projective_from_montgomery_kernel<<<blocks, threads, 0, stream>>>(
        output, input, size
    );
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

} // extern "C"
