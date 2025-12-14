/**
 * @file params.cu
 * @brief BLS12-381 Device Constant Definitions
 * 
 * =============================================================================
 * PURPOSE
 * =============================================================================
 * 
 * This file defines the __device__ __constant__ arrays for GPU constant memory.
 * Values are sourced from bls12_381_constants.h (single source of truth).
 * 
 * The header (bls12_381_params.cuh) declares these as extern. This file
 * provides the actual definitions that the device linker resolves.
 * 
 * =============================================================================
 * BUILD REQUIREMENTS
 * =============================================================================
 * 
 * This file must be compiled with CUDA separate compilation:
 *   nvcc -dc -rdc=true params.cu -o params.o
 * 
 * And linked with device linking:
 *   nvcc -dlink params.o other.o -o device_link.o
 * 
 * =============================================================================
 */

#include <cstdint>
#include "bls12_381_constants.h"

namespace bls12_381 {

// =============================================================================
// Base Field Fq Constants
// =============================================================================

__device__ __constant__ uint64_t FQ_MODULUS[BLS12_381_FP_LIMBS_64] = FQ_MODULUS_LIMBS;
__device__ __constant__ uint64_t FQ_ONE[BLS12_381_FP_LIMBS_64]     = FQ_ONE_LIMBS;
__device__ __constant__ uint64_t FQ_R2[BLS12_381_FP_LIMBS_64]      = FQ_R2_LIMBS;

// =============================================================================
// Scalar Field Fr Constants
// =============================================================================

__device__ __constant__ uint64_t FR_MODULUS[BLS12_381_FR_LIMBS_64] = FR_MODULUS_LIMBS;
__device__ __constant__ uint64_t FR_ONE[BLS12_381_FR_LIMBS_64]     = FR_ONE_LIMBS;
__device__ __constant__ uint64_t FR_R2[BLS12_381_FR_LIMBS_64]      = FR_R2_LIMBS;

} // namespace bls12_381
