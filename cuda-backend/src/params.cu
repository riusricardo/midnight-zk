/**
 * @file params.cu
 * @brief BLS12-381 constant definitions
 * 
 * This file contains the actual definitions of __device__ __constant__ arrays.
 * Headers declare them as extern to avoid ODR violations.
 */

#include <cstdint>

namespace bls12_381 {

// =============================================================================
// Base Field (Fq) Parameters - 381-bit prime
// =============================================================================

constexpr int FP_LIMBS_64 = 6;
constexpr int FR_LIMBS_64 = 4;

// Modulus p (little-endian limbs)
__device__ __constant__ uint64_t FQ_MODULUS[FP_LIMBS_64] = {
    0xb9feffffffffaaabULL,
    0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL,
    0x1a0111ea397fe69aULL
};

// R = 2^384 mod p (Montgomery one)
__device__ __constant__ uint64_t FQ_ONE[FP_LIMBS_64] = {
    0x760900000002fffdULL,
    0xebf4000bc40c0002ULL,
    0x5f48985753c758baULL,
    0x77ce585370525745ULL,
    0x5c071a97a256ec6dULL,
    0x15f65ec3fa80e493ULL
};

// R^2 = 2^768 mod p (for converting to Montgomery form)
__device__ __constant__ uint64_t FQ_R2[FP_LIMBS_64] = {
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

// Modulus r (little-endian limbs)
__device__ __constant__ uint64_t FR_MODULUS[FR_LIMBS_64] = {
    0xffffffff00000001ULL,
    0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL,
    0x73eda753299d7d48ULL
};

// R = 2^256 mod r (Montgomery one)
__device__ __constant__ uint64_t FR_ONE[FR_LIMBS_64] = {
    0x00000001fffffffeULL,
    0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL,
    0x1824b159acc5056fULL
};

// R^2 = 2^512 mod r (for converting to Montgomery form)
__device__ __constant__ uint64_t FR_R2[FR_LIMBS_64] = {
    0xc999e990f3f29c6dULL,
    0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL,
    0x0748d9d99f59ff11ULL
};

} // namespace bls12_381
