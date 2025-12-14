/**
 * @file bls12_381_constants.h
 * @brief BLS12-381 Curve Constants - Single Source of Truth
 * 
 * =============================================================================
 * AUDIT NOTICE
 * =============================================================================
 * This file contains all cryptographic constants for the BLS12-381 curve.
 * All values are defined here ONCE and consumed by both:
 *   - Host code (via constexpr arrays in bls12_381_params.cuh)
 *   - Device code (via __constant__ arrays in src/params.cu)
 * 
 * DO NOT define these constants anywhere else. Any modification to curve
 * parameters must be made in this file only.
 * 
 * Constants verified against:
 *   - BLST library (https://github.com/supranational/blst)
 *   - Arkworks (https://github.com/arkworks-rs/curves)
 *   - EIP-2537 specification
 * 
 * All multi-limb values are in LITTLE-ENDIAN order (least significant first).
 * Field elements are in Montgomery representation where noted.
 * =============================================================================
 */

#ifndef BLS12_381_CONSTANTS_H
#define BLS12_381_CONSTANTS_H

/* ============================================================================
 * Limb Configuration
 * ============================================================================ */

#define BLS12_381_FP_LIMBS_64  6   /* 384-bit base field Fq */
#define BLS12_381_FR_LIMBS_64  4   /* 256-bit scalar field Fr */

/* ============================================================================
 * Base Field Fq (381-bit prime)
 * ============================================================================
 * p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
 * 
 * This is the field over which G1 points are defined.
 * ============================================================================ */

/* Modulus p in little-endian 64-bit limbs */
#define FQ_MODULUS_L0  0xb9feffffffffaaabULL
#define FQ_MODULUS_L1  0x1eabfffeb153ffffULL
#define FQ_MODULUS_L2  0x6730d2a0f6b0f624ULL
#define FQ_MODULUS_L3  0x64774b84f38512bfULL
#define FQ_MODULUS_L4  0x4b1ba7b6434bacd7ULL
#define FQ_MODULUS_L5  0x1a0111ea397fe69aULL

/* Montgomery constant: -p^{-1} mod 2^64 */
#define FQ_INV_VALUE   0x89f3fffcfffcfffdULL

/* R = 2^384 mod p (Montgomery one)
 * This is the multiplicative identity in Montgomery form */
#define FQ_ONE_L0      0x760900000002fffdULL
#define FQ_ONE_L1      0xebf4000bc40c0002ULL
#define FQ_ONE_L2      0x5f48985753c758baULL
#define FQ_ONE_L3      0x77ce585370525745ULL
#define FQ_ONE_L4      0x5c071a97a256ec6dULL
#define FQ_ONE_L5      0x15f65ec3fa80e493ULL

/* R^2 = 2^768 mod p (for converting integers to Montgomery form)
 * To convert x to Montgomery form: mont(x) = x * R^2 * R^{-1} mod p = x * R mod p */
#define FQ_R2_L0       0xf4df1f341c341746ULL
#define FQ_R2_L1       0x0a76e6a609d104f1ULL
#define FQ_R2_L2       0x8de5476c4c95b6d5ULL
#define FQ_R2_L3       0x67eb88a9939d83c0ULL
#define FQ_R2_L4       0x9a793e85b519952dULL
#define FQ_R2_L5       0x11988fe592cae3aaULL

/* ============================================================================
 * Scalar Field Fr (255-bit prime)
 * ============================================================================
 * r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
 * 
 * This is the order of the G1 and G2 groups (number of points on the curve).
 * ============================================================================ */

/* Modulus r in little-endian 64-bit limbs */
#define FR_MODULUS_L0  0xffffffff00000001ULL
#define FR_MODULUS_L1  0x53bda402fffe5bfeULL
#define FR_MODULUS_L2  0x3339d80809a1d805ULL
#define FR_MODULUS_L3  0x73eda753299d7d48ULL

/* Montgomery constant: -r^{-1} mod 2^64 */
#define FR_INV_VALUE   0xfffffffeffffffffULL

/* R = 2^256 mod r (Montgomery one) */
#define FR_ONE_L0      0x00000001fffffffeULL
#define FR_ONE_L1      0x5884b7fa00034802ULL
#define FR_ONE_L2      0x998c4fefecbc4ff5ULL
#define FR_ONE_L3      0x1824b159acc5056fULL

/* R^2 = 2^512 mod r (for converting integers to Montgomery form) */
#define FR_R2_L0       0xc999e990f3f29c6dULL
#define FR_R2_L1       0x2b6cedcb87925c23ULL
#define FR_R2_L2       0x05d314967254398fULL
#define FR_R2_L3       0x0748d9d99f59ff11ULL

/* ============================================================================
 * G1 Curve Parameters
 * ============================================================================
 * G1: y^2 = x^3 + 4 over Fq
 * ============================================================================ */

/* Curve coefficient b = 4 in Montgomery form */
#define G1_B_L0        0xaa270000000cfff3ULL
#define G1_B_L1        0x53cc0032fc34000aULL
#define G1_B_L2        0x478fe97a6b0a807fULL
#define G1_B_L3        0xb1d37ebee6ba24d7ULL
#define G1_B_L4        0x8ec9733bbf78ab2fULL
#define G1_B_L5        0x09d645513d83de7eULL

/* Generator point G1 - x coordinate in Montgomery form */
#define G1_GEN_X_L0    0x5cb38790fd530c16ULL
#define G1_GEN_X_L1    0x7817fc679976fff5ULL
#define G1_GEN_X_L2    0x154f95c7143ba1c1ULL
#define G1_GEN_X_L3    0xf0ae6acdf3d0e747ULL
#define G1_GEN_X_L4    0xedce6ecc21dbf440ULL
#define G1_GEN_X_L5    0x120177419e0bfb75ULL

/* Generator point G1 - y coordinate in Montgomery form */
#define G1_GEN_Y_L0    0xbaac93d50ce72271ULL
#define G1_GEN_Y_L1    0x8c22631a7918fd8eULL
#define G1_GEN_Y_L2    0xdd595f13570725ceULL
#define G1_GEN_Y_L3    0x51ac582950405194ULL
#define G1_GEN_Y_L4    0x0e1c8c3fad0059c0ULL
#define G1_GEN_Y_L5    0x0bbc3efc5008a26aULL

/* ============================================================================
 * Convenience Macros for Array Initialization
 * ============================================================================ */

#define FQ_MODULUS_LIMBS  { FQ_MODULUS_L0, FQ_MODULUS_L1, FQ_MODULUS_L2, \
                            FQ_MODULUS_L3, FQ_MODULUS_L4, FQ_MODULUS_L5 }

#define FQ_ONE_LIMBS      { FQ_ONE_L0, FQ_ONE_L1, FQ_ONE_L2, \
                            FQ_ONE_L3, FQ_ONE_L4, FQ_ONE_L5 }

#define FQ_R2_LIMBS       { FQ_R2_L0, FQ_R2_L1, FQ_R2_L2, \
                            FQ_R2_L3, FQ_R2_L4, FQ_R2_L5 }

#define FR_MODULUS_LIMBS  { FR_MODULUS_L0, FR_MODULUS_L1, \
                            FR_MODULUS_L2, FR_MODULUS_L3 }

#define FR_ONE_LIMBS      { FR_ONE_L0, FR_ONE_L1, FR_ONE_L2, FR_ONE_L3 }

#define FR_R2_LIMBS       { FR_R2_L0, FR_R2_L1, FR_R2_L2, FR_R2_L3 }

#define G1_B_LIMBS        { G1_B_L0, G1_B_L1, G1_B_L2, \
                            G1_B_L3, G1_B_L4, G1_B_L5 }

#define G1_GEN_X_LIMBS    { G1_GEN_X_L0, G1_GEN_X_L1, G1_GEN_X_L2, \
                            G1_GEN_X_L3, G1_GEN_X_L4, G1_GEN_X_L5 }

#define G1_GEN_Y_LIMBS    { G1_GEN_Y_L0, G1_GEN_Y_L1, G1_GEN_Y_L2, \
                            G1_GEN_Y_L3, G1_GEN_Y_L4, G1_GEN_Y_L5 }

#endif /* BLS12_381_CONSTANTS_H */
