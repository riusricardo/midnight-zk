/**
 * @file field.cuh
 * @brief BLS12-381 Field element types and arithmetic
 * 
 * Implements both Fq (base field, 381-bit) and Fr (scalar field, 255-bit)
 * using Montgomery representation for efficient modular arithmetic.
 */

#pragma once

#include "bls12_381_params.cuh"
#include <cuda_runtime.h>

namespace bls12_381 {

// =============================================================================
// Forward declarations
// =============================================================================

template<typename Config> struct Field;
struct fp_config;
struct fq_config;

// =============================================================================
// Field configuration traits - using inline device functions for constants
// =============================================================================

struct fp_config {
    static constexpr int LIMBS = FR_LIMBS_64;
    static constexpr int NBITS = 255;
    static constexpr uint64_t INV = FR_INV;
    
    __device__ __forceinline__ static uint64_t modulus(int i) {
        return FR_MODULUS[i];
    }
    
    __device__ __forceinline__ static uint64_t one(int i) {
        return FR_ONE[i];
    }
    
    __device__ __forceinline__ static uint64_t r2(int i) {
        return FR_R2[i];
    }
    
    __host__ static uint64_t modulus_host(int i) {
        return FR_MODULUS_HOST[i];
    }
    
    __host__ static uint64_t one_host(int i) {
        return FR_ONE_HOST[i];
    }
    
    __host__ static uint64_t r2_host(int i) {
        return FR_R2_HOST[i];
    }
};

struct fq_config {
    static constexpr int LIMBS = FP_LIMBS_64;
    static constexpr int NBITS = 381;
    static constexpr uint64_t INV = FQ_INV;
    
    __device__ __forceinline__ static uint64_t modulus(int i) {
        return FQ_MODULUS[i];
    }
    
    __device__ __forceinline__ static uint64_t one(int i) {
        return FQ_ONE[i];
    }
    
    __device__ __forceinline__ static uint64_t r2(int i) {
        return FQ_R2[i];
    }
    
    __host__ static uint64_t modulus_host(int i) {
        return FQ_MODULUS_HOST[i];
    }
    
    __host__ static uint64_t one_host(int i) {
        return FQ_ONE_HOST[i];
    }
    
    __host__ static uint64_t r2_host(int i) {
        return FQ_R2_HOST[i];
    }
};

// =============================================================================
// Field element storage
// =============================================================================

template<typename Config>
struct alignas(32) Field {
    static constexpr int LIMBS = Config::LIMBS;
    uint64_t limbs[LIMBS];

    // Default constructor (zero)
    __host__ __device__ Field() {
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            limbs[i] = 0;
        }
    }

    // Constructor from limbs array
    __host__ __device__ explicit Field(const uint64_t* data) {
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            limbs[i] = data[i];
        }
    }

    // Zero element
    __host__ __device__ static Field zero() {
        return Field();
    }

    // One element (Montgomery form) - device version
    __device__ static Field one() {
        Field result;
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::one(i);
        }
        return result;
    }
    
    // One element - host version
    __host__ static Field one_host() {
        Field result;
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::one_host(i);
        }
        return result;
    }
    
    // R^2 for Montgomery conversion - device version
    __device__ static Field R_SQUARED() {
        Field result;
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            result.limbs[i] = Config::r2(i);
        }
        return result;
    }
    
    // Create from integer
    __device__ static Field from_int(uint64_t val) {
        Field result;
        result.limbs[0] = val;
        for (int i = 1; i < LIMBS; i++) {
            result.limbs[i] = 0;
        }
        // Convert to Montgomery form
        return result * R_SQUARED();
    }

    // Check if zero
    __host__ __device__ bool is_zero() const {
        uint64_t acc = 0;
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            acc |= limbs[i];
        }
        return acc == 0;
    }

    // Equality
    __host__ __device__ bool operator==(const Field& other) const {
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            if (limbs[i] != other.limbs[i]) return false;
        }
        return true;
    }

    __host__ __device__ bool operator!=(const Field& other) const {
        return !(*this == other);
    }
};

// Type aliases
using Fr = Field<fp_config>;  // Scalar field
using Fq = Field<fq_config>;  // Base field

// =============================================================================
// Montgomery arithmetic - Device functions
// =============================================================================

/**
 * @brief Add two field elements: result = a + b mod p
 */
template<typename Config>
__device__ __forceinline__ void field_add(
    Field<Config>& result,
    const Field<Config>& a,
    const Field<Config>& b
) {
    constexpr int LIMBS = Config::LIMBS;
    
    uint64_t carry = 0;
    uint64_t temp[LIMBS];
    
    // First addition: a + b
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t sum = ai + bi;
        uint64_t new_carry = (sum < ai) ? 1ULL : 0ULL;
        sum += carry;
        new_carry += (sum < carry) ? 1ULL : 0ULL;
        temp[i] = sum;
        carry = new_carry;
    }
    
    // Conditional subtraction if >= modulus
    uint64_t borrow = 0;
    uint64_t reduced[LIMBS];
    
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        uint64_t t = temp[i];
        uint64_t diff = t - mod_i;
        uint64_t new_borrow = (t < mod_i) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - borrow;
        new_borrow += (diff < borrow) ? 1ULL : 0ULL;
        reduced[i] = diff2;
        borrow = new_borrow;
    }
    
    // Select result based on whether subtraction underflowed
    bool use_reduced = (carry != 0) || (borrow == 0);
    
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        result.limbs[i] = use_reduced ? reduced[i] : temp[i];
    }
}

/**
 * @brief Subtract two field elements: result = a - b mod p
 */
template<typename Config>
__device__ __forceinline__ void field_sub(
    Field<Config>& result,
    const Field<Config>& a,
    const Field<Config>& b
) {
    constexpr int LIMBS = Config::LIMBS;
    
    uint64_t borrow = 0;
    uint64_t temp[LIMBS];
    
    // Subtraction: a - b
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t diff = ai - bi;
        uint64_t new_borrow = (ai < bi) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - borrow;
        new_borrow += (diff < borrow) ? 1ULL : 0ULL;
        temp[i] = diff2;
        borrow = new_borrow;
    }
    
    // If borrow, add modulus
    if (borrow) {
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < LIMBS; i++) {
            uint64_t t = temp[i];
            uint64_t mod_i = Config::modulus(i);
            uint64_t sum = t + mod_i;
            uint64_t new_carry = (sum < t) ? 1ULL : 0ULL;
            sum += carry;
            new_carry += (sum < carry) ? 1ULL : 0ULL;
            temp[i] = sum;
            carry = new_carry;
        }
    }
    
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        result.limbs[i] = temp[i];
    }
}

/**
 * @brief Montgomery multiplication: result = a * b * R^{-1} mod p
 * 
 * Uses the CIOS (Coarsely Integrated Operand Scanning) algorithm
 * optimized for GPU execution.
 */
template<typename Config>
__device__ __forceinline__ void field_mul(
    Field<Config>& result,
    const Field<Config>& a,
    const Field<Config>& b
) {
    constexpr int LIMBS = Config::LIMBS;
    const uint64_t inv = Config::INV;
    
    uint64_t t[LIMBS + 2] = {0};
    
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        // Multiply accumulate: t += a[i] * b
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < LIMBS; j++) {
            // t[j] += a[i] * b[j] + carry
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * b.limbs[j];
            prod += t[j];
            prod += carry;
            t[j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[LIMBS] += carry;
        t[LIMBS + 1] = (t[LIMBS] < carry) ? 1 : 0;
        
        // Montgomery reduction step
        uint64_t m = t[0] * inv;
        
        carry = 0;
        #pragma unroll
        for (int j = 0; j < LIMBS; j++) {
            unsigned __int128 prod = (unsigned __int128)m * Config::modulus(j);
            prod += t[j];
            prod += carry;
            if (j > 0) t[j - 1] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[LIMBS - 1] = t[LIMBS] + carry;
        t[LIMBS] = t[LIMBS + 1] + ((t[LIMBS - 1] < carry) ? 1 : 0);
        t[LIMBS + 1] = 0;
    }
    
    // Final reduction
    uint64_t borrow = 0;
    uint64_t reduced[LIMBS];
    
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        uint64_t diff = t[i] - mod_i - borrow;
        borrow = (t[i] < mod_i + borrow) ? 1 : 0;
        reduced[i] = diff;
    }
    
    bool use_reduced = (t[LIMBS] != 0) || (borrow == 0);
    
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        result.limbs[i] = use_reduced ? reduced[i] : t[i];
    }
}

/**
 * @brief Montgomery squaring: result = a^2 * R^{-1} mod p
 */
template<typename Config>
__device__ __forceinline__ void field_sqr(
    Field<Config>& result,
    const Field<Config>& a
) {
    // For now, use multiplication. Can be optimized with dedicated squaring.
    field_mul(result, a, a);
}

/**
 * @brief Negate field element: result = -a mod p
 */
template<typename Config>
__device__ __forceinline__ void field_neg(
    Field<Config>& result,
    const Field<Config>& a
) {
    if (a.is_zero()) {
        result = a;
        return;
    }
    
    constexpr int LIMBS = Config::LIMBS;
    
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        uint64_t diff = mod_i - a.limbs[i] - borrow;
        borrow = (mod_i < a.limbs[i] + borrow) ? 1 : 0;
        result.limbs[i] = diff;
    }
}

/**
 * @brief Modular inversion using Fermat's little theorem: a^{-1} = a^{p-2} mod p
 * 
 * Uses binary method for exponentiation.
 */
template<typename Config>
__device__ void field_inv(
    Field<Config>& result,
    const Field<Config>& a
) {
    // p - 2 for the exponent
    constexpr int LIMBS = Config::LIMBS;
    
    // Compute exponent = p - 2
    uint64_t exp[LIMBS];
    uint64_t borrow = 0;
    
    // First subtract 2 from modulus
    uint64_t mod_0 = Config::modulus(0);
    exp[0] = mod_0 - 2;
    borrow = (mod_0 < 2) ? 1 : 0;
    
    #pragma unroll
    for (int i = 1; i < LIMBS; i++) {
        uint64_t mod_i = Config::modulus(i);
        exp[i] = mod_i - borrow;
        borrow = (mod_i < borrow) ? 1 : 0;
    }
    
    // Binary exponentiation
    Field<Config> base = a;
    result = Field<Config>::one();
    
    for (int i = 0; i < LIMBS; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1) {
                field_mul(result, result, base);
            }
            field_sqr(base, base);
        }
    }
}

/**
 * @brief Convert from standard to Montgomery form: result = a * R mod p
 */
template<typename Config>
__device__ __forceinline__ void field_to_montgomery(
    Field<Config>& result,
    const Field<Config>& a
) {
    Field<Config> r2 = Field<Config>::R_SQUARED();
    field_mul(result, a, r2);
}

/**
 * @brief Convert from Montgomery to standard form: result = a * R^{-1} mod p
 */
template<typename Config>
__device__ __forceinline__ void field_from_montgomery(
    Field<Config>& result,
    const Field<Config>& a
) {
    Field<Config> one_raw;
    one_raw.limbs[0] = 1;
    for (int i = 1; i < Config::LIMBS; i++) {
        one_raw.limbs[i] = 0;
    }
    field_mul(result, a, one_raw);
}

// =============================================================================
// Convenience wrapper functions (return value versions)
// =============================================================================

/**
 * @brief Negate field element (returns value)
 */
template<typename Config>
__device__ __forceinline__ Field<Config> field_neg(const Field<Config>& a) {
    Field<Config> result;
    field_neg(result, a);
    return result;
}

/**
 * @brief Invert field element (returns value)
 */
template<typename Config>
__device__ __forceinline__ Field<Config> field_inv(const Field<Config>& a) {
    Field<Config> result;
    field_inv(result, a);
    return result;
}

// =============================================================================
// Operator overloads for cleaner syntax
// =============================================================================

template<typename Config>
__device__ __forceinline__ Field<Config> operator+(
    const Field<Config>& a,
    const Field<Config>& b
) {
    Field<Config> result;
    field_add(result, a, b);
    return result;
}

template<typename Config>
__device__ __forceinline__ Field<Config> operator-(
    const Field<Config>& a,
    const Field<Config>& b
) {
    Field<Config> result;
    field_sub(result, a, b);
    return result;
}

template<typename Config>
__device__ __forceinline__ Field<Config> operator*(
    const Field<Config>& a,
    const Field<Config>& b
) {
    Field<Config> result;
    field_mul(result, a, b);
    return result;
}

template<typename Config>
__device__ __forceinline__ Field<Config> operator-(const Field<Config>& a) {
    Field<Config> result;
    field_neg(result, a);
    return result;
}

} // namespace bls12_381
