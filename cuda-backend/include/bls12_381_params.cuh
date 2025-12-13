/**
 * @file bls12_381_params.cuh
 * @brief BLS12-381 curve parameters and constants
 * 
 * Constants extracted from BLST and verified against Icicle binary analysis.
 * All values are in Montgomery representation where applicable.
 * 
 * Device constant arrays are declared extern here and defined in params.cu.
 */

#pragma once

#include <cstdint>

namespace bls12_381 {

// =============================================================================
// Limb configuration
// =============================================================================

// 384-bit field elements use 6 x 64-bit limbs (or 12 x 32-bit for GPU efficiency)
constexpr int FP_LIMBS_64 = 6;
constexpr int FP_LIMBS_32 = 12;

// 256-bit scalar field uses 4 x 64-bit limbs (or 8 x 32-bit)
constexpr int FR_LIMBS_64 = 4;
constexpr int FR_LIMBS_32 = 8;

// =============================================================================
// Base Field (Fq) Parameters - 381-bit prime
// =============================================================================
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

// Modulus p (little-endian limbs) - defined in params.cu
extern __device__ __constant__ uint64_t FQ_MODULUS[FP_LIMBS_64];

// Host-side copy for CPU code
constexpr uint64_t FQ_MODULUS_HOST[FP_LIMBS_64] = {
    0xb9feffffffffaaabULL,
    0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL,
    0x1a0111ea397fe69aULL
};

// Montgomery constant: -p^{-1} mod 2^64
constexpr uint64_t FQ_INV = 0x89f3fffcfffcfffdULL;

// R = 2^384 mod p (Montgomery one) - defined in params.cu
extern __device__ __constant__ uint64_t FQ_ONE[FP_LIMBS_64];

constexpr uint64_t FQ_ONE_HOST[FP_LIMBS_64] = {
    0x760900000002fffdULL,
    0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL,
    0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL,
    0x15f65ec3fa80e493ULL
};

// R^2 = 2^768 mod p (for converting to Montgomery form) - defined in params.cu
extern __device__ __constant__ uint64_t FQ_R2[FP_LIMBS_64];

constexpr uint64_t FQ_R2_HOST[FP_LIMBS_64] = {
    0xf4df1f341c341746ULL,
    0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL,
    0x67eb88a9939d83c0ULL,
    0x9a793e85b519952dULL,
    0x11988fe592cae3aaULL
};

// =============================================================================
// Scalar Field (Fr) Parameters - 255-bit prime
// =============================================================================
// r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

// Modulus r (little-endian limbs) - defined in params.cu
extern __device__ __constant__ uint64_t FR_MODULUS[FR_LIMBS_64];

constexpr uint64_t FR_MODULUS_HOST[FR_LIMBS_64] = {
    0xffffffff00000001ULL,
    0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL,
    0x73eda753299d7d48ULL
};

// Montgomery constant: -r^{-1} mod 2^64
constexpr uint64_t FR_INV = 0xfffffffeffffffffULL;

// R = 2^256 mod r (Montgomery one) - defined in params.cu
extern __device__ __constant__ uint64_t FR_ONE[FR_LIMBS_64];

constexpr uint64_t FR_ONE_HOST[FR_LIMBS_64] = {
    0x00000001fffffffeULL,
    0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL,
    0x1824b159acc5056fULL
};

// R^2 = 2^512 mod r (for converting to Montgomery form) - defined in params.cu
extern __device__ __constant__ uint64_t FR_R2[FR_LIMBS_64];

constexpr uint64_t FR_R2_HOST[FR_LIMBS_64] = {
    0xc999e990f3f29c6dULL,
    0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL,
    0x0748d9d99f59ff11ULL
};

// =============================================================================
// Curve Parameters
// =============================================================================

// G1 curve: y^2 = x^3 + 4
// Coefficient B in Montgomery form
constexpr uint64_t G1_B[FP_LIMBS_64] = {
    0xaa270000000cfff3ULL,
    0x53cc0032fc34000aULL,
    0x478fe97a6b0a807fULL,
    0xb1d37ebee6ba24d7ULL,
    0x8ec9733bbf78ab2fULL,
    0x09d645513d83de7eULL
};

// Generator point for G1 (in Montgomery form)
constexpr uint64_t G1_GENERATOR_X[FP_LIMBS_64] = {
    0x5cb38790fd530c16ULL,
    0x7817fc679976fff5ULL,
    0x154f95c7143ba1c1ULL,
    0xf0ae6acdf3d0e747ULL,
    0xedce6ecc21dbf440ULL,
    0x120177419e0bfb75ULL
};

constexpr uint64_t G1_GENERATOR_Y[FP_LIMBS_64] = {
    0xbaac93d50ce72271ULL,
    0x8c22631a7918fd8eULL,
    0xdd595f13570725ceULL,
    0x51ac582950405194ULL,
    0x0e1c8c3fad0059c0ULL,
    0x0bbc3efc5008a26aULL
};

// =============================================================================
// NTT Parameters  
// =============================================================================

// Maximum NTT size: 2^32 (determined by scalar field order r-1 = 2^32 * m)
constexpr int MAX_NTT_LOG_SIZE = 32;

// Primitive 2^32-th root of unity in Fr (Montgomery form)
// omega = generator^((r-1)/2^32)
constexpr uint64_t FR_OMEGA[FR_LIMBS_64] = {
    0xb9b58d8c5f0e466aULL,
    0x5b1b4c801819d7ecULL,
    0x0af53ae352a31e64ULL,
    0x5bf3adda19e9b27bULL
};

} // namespace bls12_381
