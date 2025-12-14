/**
 * @file bls12_381_params.cuh
 * @brief BLS12-381 CUDA type definitions and constant declarations
 * 
 * =============================================================================
 * Architecture
 * =============================================================================
 * 
 * This file provides:
 *   1. Host-side constexpr arrays - for CPU code paths
 *   2. Device extern declarations - for GPU code (defined in src/params.cu)
 * 
 * All constant VALUES are defined in bls12_381_constants.h (single source of
 * truth). This file merely creates the appropriate C++/CUDA bindings.
 * 
 * Build requires CUDA separate compilation:
 *   nvcc -dc -rdc=true  (compile)
 *   nvcc -dlink         (device link)
 * 
 * =============================================================================
 */

#pragma once

#include <cstdint>
#include "bls12_381_constants.h"

namespace bls12_381 {

// =============================================================================
// Limb Configuration
// =============================================================================

constexpr int FP_LIMBS_64 = BLS12_381_FP_LIMBS_64;  // 384-bit base field
constexpr int FP_LIMBS_32 = FP_LIMBS_64 * 2;
constexpr int FR_LIMBS_64 = BLS12_381_FR_LIMBS_64;  // 256-bit scalar field
constexpr int FR_LIMBS_32 = FR_LIMBS_64 * 2;

// =============================================================================
// Base Field Fq - Host Constants
// =============================================================================

constexpr uint64_t FQ_MODULUS_HOST[FP_LIMBS_64] = FQ_MODULUS_LIMBS;
constexpr uint64_t FQ_ONE_HOST[FP_LIMBS_64]     = FQ_ONE_LIMBS;
constexpr uint64_t FQ_R2_HOST[FP_LIMBS_64]      = FQ_R2_LIMBS;
constexpr uint64_t FQ_INV                       = FQ_INV_VALUE;

// =============================================================================
// Base Field Fq - Device Constants (defined in src/params.cu)
// =============================================================================

extern __device__ __constant__ uint64_t FQ_MODULUS[FP_LIMBS_64];
extern __device__ __constant__ uint64_t FQ_ONE[FP_LIMBS_64];
extern __device__ __constant__ uint64_t FQ_R2[FP_LIMBS_64];

// =============================================================================
// Scalar Field Fr - Host Constants
// =============================================================================

constexpr uint64_t FR_MODULUS_HOST[FR_LIMBS_64] = FR_MODULUS_LIMBS;
constexpr uint64_t FR_ONE_HOST[FR_LIMBS_64]     = FR_ONE_LIMBS;
constexpr uint64_t FR_R2_HOST[FR_LIMBS_64]      = FR_R2_LIMBS;
constexpr uint64_t FR_INV                       = FR_INV_VALUE;

// =============================================================================
// Scalar Field Fr - Device Constants (defined in src/params.cu)
// =============================================================================

extern __device__ __constant__ uint64_t FR_MODULUS[FR_LIMBS_64];
extern __device__ __constant__ uint64_t FR_ONE[FR_LIMBS_64];
extern __device__ __constant__ uint64_t FR_R2[FR_LIMBS_64];

// =============================================================================
// G1 Curve Parameters - Host Constants
// =============================================================================

constexpr uint64_t G1_B[FP_LIMBS_64]           = G1_B_LIMBS;
constexpr uint64_t G1_GENERATOR_X[FP_LIMBS_64] = G1_GEN_X_LIMBS;
constexpr uint64_t G1_GENERATOR_Y[FP_LIMBS_64] = G1_GEN_Y_LIMBS;

// =============================================================================
// G2 Curve Parameters - Host Constants
// =============================================================================
// G2 coordinates are Fq2 elements: each has c0 (real) and c1 (imaginary) parts

// Curve coefficient b' = 4(1+u) in Fq2
constexpr uint64_t G2_B_C0[FP_LIMBS_64]           = G2_B_C0_LIMBS;
constexpr uint64_t G2_B_C1[FP_LIMBS_64]           = G2_B_C1_LIMBS;

// Generator x coordinate (Fq2)
constexpr uint64_t G2_GENERATOR_X_C0[FP_LIMBS_64] = G2_GEN_X_C0_LIMBS;
constexpr uint64_t G2_GENERATOR_X_C1[FP_LIMBS_64] = G2_GEN_X_C1_LIMBS;

// Generator y coordinate (Fq2)
constexpr uint64_t G2_GENERATOR_Y_C0[FP_LIMBS_64] = G2_GEN_Y_C0_LIMBS;
constexpr uint64_t G2_GENERATOR_Y_C1[FP_LIMBS_64] = G2_GEN_Y_C1_LIMBS;

// =============================================================================
// NTT Parameters
// =============================================================================

// Maximum NTT size: 2^32 (determined by scalar field order r-1 = 2^32 * m)
constexpr int MAX_NTT_LOG_SIZE = 32;

// Primitive 2^32-th root of unity in Fr (Montgomery form)
// omega = generator^((r-1)/2^32)
// TODO: Move to bls12_381_constants.h after verification
constexpr uint64_t FR_OMEGA[FR_LIMBS_64] = {
    0xb9b58d8c5f0e466aULL,
    0x5b1b4c801819d7ecULL,
    0x0af53ae352a31e64ULL,
    0x5bf3adda19e9b27bULL
};

} // namespace bls12_381
