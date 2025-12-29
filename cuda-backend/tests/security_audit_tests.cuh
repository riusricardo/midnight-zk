/**
 * @file security_audit_tests.cuh
 * @brief Security Audit Test Framework for BLS12-381 CUDA Backend
 * 
 * This header provides a comprehensive test framework designed to pass
 * rigorous security audits for cryptographic implementations.
 * 
 * SECURITY AUDIT CHECKLIST:
 * ========================
 * 1. Known Answer Tests (KAT) with verified test vectors
 * 2. Algebraic property verification (group laws, field axioms)
 * 3. Edge case handling (zero, identity, boundary values)
 * 4. Constant-time behavior verification
 * 5. Montgomery form correctness
 * 6. Cross-implementation validation
 * 7. Memory safety (no buffer overflows, proper bounds checking)
 * 8. Error handling (graceful failure on invalid inputs)
 * 
 * TEST VECTOR SOURCES:
 * ====================
 * - BLS12-381 specification (EIP-2537)
 * - BLST library test vectors
 * - Arkworks reference implementation
 * - Zcash Sapling specification
 */

#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <functional>

#include "field.cuh"
#include "point.cuh"
#include "icicle_types.cuh"
#include "ntt.cuh"
#include "msm.cuh"

namespace security_tests {

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// Test Framework Infrastructure
// =============================================================================

#define SECURITY_CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        return TestResult::CUDA_ERROR; \
    } \
} while(0)

enum class TestResult {
    PASSED,
    FAILED,
    SKIPPED,
    CUDA_ERROR,
    TIMEOUT
};

struct TestCase {
    std::string name;
    std::string category;
    std::function<TestResult()> test_fn;
    bool critical;  // If critical test fails, abort remaining tests
};

class SecurityTestSuite {
public:
    std::vector<TestCase> tests;
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    int cuda_errors = 0;
    std::vector<std::string> failed_tests;
    std::vector<std::string> critical_failures;
    
    void add_test(const std::string& name, const std::string& category,
                  std::function<TestResult()> fn, bool critical = false) {
        tests.push_back({name, category, fn, critical});
    }
    
    bool run_all() {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     BLS12-381 CUDA Security Audit Test Suite               ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;
        
        std::string current_category;
        
        for (const auto& tc : tests) {
            if (tc.category != current_category) {
                current_category = tc.category;
                std::cout << "\n═══ " << current_category << " ═══" << std::endl;
            }
            
            std::cout << "  [" << (tc.critical ? "CRIT" : "TEST") << "] " 
                      << tc.name << "... " << std::flush;
            
            auto start = std::chrono::high_resolution_clock::now();
            TestResult result = tc.test_fn();
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            switch (result) {
                case TestResult::PASSED:
                    std::cout << "✓ PASSED (" << std::fixed << std::setprecision(2) << ms << "ms)" << std::endl;
                    passed++;
                    break;
                case TestResult::FAILED:
                    std::cout << "✗ FAILED" << std::endl;
                    failed++;
                    failed_tests.push_back(tc.name);
                    if (tc.critical) {
                        critical_failures.push_back(tc.name);
                    }
                    break;
                case TestResult::SKIPPED:
                    std::cout << "○ SKIPPED" << std::endl;
                    skipped++;
                    break;
                case TestResult::CUDA_ERROR:
                    std::cout << "⚠ CUDA ERROR" << std::endl;
                    cuda_errors++;
                    break;
                case TestResult::TIMEOUT:
                    std::cout << "⏱ TIMEOUT" << std::endl;
                    failed++;
                    failed_tests.push_back(tc.name + " (timeout)");
                    break;
            }
            
            // Abort on critical failure
            if (tc.critical && result != TestResult::PASSED) {
                std::cerr << "\n  ⚠ CRITICAL TEST FAILED - Aborting remaining tests" << std::endl;
                break;
            }
        }
        
        print_summary();
        return failed == 0 && cuda_errors == 0;
    }
    
    void print_summary() {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                      TEST SUMMARY                          ║" << std::endl;
        std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║  Passed:      " << std::setw(4) << passed << "                                        ║" << std::endl;
        std::cout << "║  Failed:      " << std::setw(4) << failed << "                                        ║" << std::endl;
        std::cout << "║  Skipped:     " << std::setw(4) << skipped << "                                        ║" << std::endl;
        std::cout << "║  CUDA Errors: " << std::setw(4) << cuda_errors << "                                        ║" << std::endl;
        std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
        
        if (failed == 0 && cuda_errors == 0) {
            std::cout << "║  ✓ ALL TESTS PASSED - Ready for Security Audit            ║" << std::endl;
        } else {
            std::cout << "║  ✗ TESTS FAILED - NOT Ready for Security Audit            ║" << std::endl;
            if (!failed_tests.empty()) {
                std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
                std::cout << "║  Failed Tests:                                             ║" << std::endl;
                for (const auto& name : failed_tests) {
                    std::cout << "║    - " << std::left << std::setw(52) << name << " ║" << std::endl;
                }
            }
        }
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

inline bool limbs_equal(const uint64_t* a, const uint64_t* b, int n) {
    return memcmp(a, b, n * sizeof(uint64_t)) == 0;
}

inline std::string limbs_to_hex(const uint64_t* limbs, int n) {
    std::ostringstream oss;
    oss << "0x";
    for (int i = n - 1; i >= 0; i--) {
        oss << std::hex << std::setfill('0') << std::setw(16) << limbs[i];
    }
    return oss.str();
}

inline void print_fr_debug(const char* name, const Fr& val) {
    std::cout << name << " = " << limbs_to_hex(val.limbs, Fr::LIMBS) << std::endl;
}

inline void print_fq_debug(const char* name, const Fq& val) {
    std::cout << name << " = " << limbs_to_hex(val.limbs, Fq::LIMBS) << std::endl;
}

// =============================================================================
// Host-side Montgomery Arithmetic for Test Vector Generation
// =============================================================================

struct uint256_t {
    uint64_t limbs[4];
    
    uint256_t() { memset(limbs, 0, sizeof(limbs)); }
    
    explicit uint256_t(uint64_t val) {
        limbs[0] = val;
        limbs[1] = limbs[2] = limbs[3] = 0;
    }
    
    explicit uint256_t(const uint64_t* data) {
        memcpy(limbs, data, sizeof(limbs));
    }
    
    bool operator>=(const uint256_t& other) const {
        for (int i = 3; i >= 0; i--) {
            if (limbs[i] > other.limbs[i]) return true;
            if (limbs[i] < other.limbs[i]) return false;
        }
        return true;
    }
    
    bool operator==(const uint256_t& other) const {
        return memcmp(limbs, other.limbs, sizeof(limbs)) == 0;
    }
    
    bool is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }
};

inline uint256_t sub256(const uint256_t& a, const uint256_t& b, uint64_t& borrow_out) {
    uint256_t result;
    __uint128_t borrow = 0;
    
    for (int i = 0; i < 4; i++) {
        __uint128_t diff = (__uint128_t)a.limbs[i] - b.limbs[i] - borrow;
        result.limbs[i] = (uint64_t)diff;
        borrow = (diff >> 64) ? 1 : 0;
    }
    borrow_out = borrow;
    return result;
}

inline void mul256_full(const uint256_t& a, const uint256_t& b, uint64_t result[8]) {
    memset(result, 0, 8 * sizeof(uint64_t));
    
    for (int i = 0; i < 4; i++) {
        __uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a.limbs[i] * b.limbs[j];
            __uint128_t sum = prod + result[i + j] + carry;
            result[i + j] = (uint64_t)sum;
            carry = sum >> 64;
        }
        result[i + 4] = (uint64_t)carry;
    }
}

inline uint256_t montgomery_reduce_host(const uint64_t T[8], const uint256_t& modulus, uint64_t inv) {
    uint64_t t[9];
    memcpy(t, T, 8 * sizeof(uint64_t));
    t[8] = 0;
    
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * inv;
        
        __uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)m * modulus.limbs[j] + t[i + j] + carry;
            t[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        
        for (int j = i + 4; j < 9; j++) {
            __uint128_t sum = (__uint128_t)t[j] + carry;
            t[j] = (uint64_t)sum;
            carry = sum >> 64;
            if (carry == 0) break;
        }
    }
    
    uint256_t result;
    memcpy(result.limbs, &t[4], 4 * sizeof(uint64_t));
    
    if (result >= modulus) {
        uint64_t borrow;
        result = sub256(result, modulus, borrow);
    }
    
    return result;
}

inline uint256_t montgomery_mul_host(const uint256_t& a, const uint256_t& b, 
                                     const uint256_t& modulus, uint64_t inv) {
    uint64_t T[8];
    mul256_full(a, b, T);
    return montgomery_reduce_host(T, modulus, inv);
}

inline Fr to_montgomery_fr(const uint256_t& a) {
    uint256_t modulus(FR_MODULUS_HOST);
    uint256_t r2(FR_R2_HOST);
    uint256_t result = montgomery_mul_host(a, r2, modulus, FR_INV);
    
    Fr fr;
    memcpy(fr.limbs, result.limbs, sizeof(fr.limbs));
    return fr;
}

inline uint256_t from_montgomery_fr(const Fr& a) {
    uint256_t modulus(FR_MODULUS_HOST);
    uint256_t val(a.limbs);
    uint256_t one(1ULL);
    return montgomery_mul_host(val, one, modulus, FR_INV);
}

inline Fr random_fr_montgomery(std::mt19937_64& rng) {
    uint256_t val;
    uint256_t modulus(FR_MODULUS_HOST);
    
    do {
        for (int i = 0; i < 4; i++) {
            val.limbs[i] = rng();
        }
        val.limbs[3] &= 0x7fffffffffffffffULL;
    } while (val >= modulus);
    
    return to_montgomery_fr(val);
}

inline Fr random_fr_nonzero(std::mt19937_64& rng) {
    Fr result;
    do {
        result = random_fr_montgomery(rng);
    } while (result.is_zero());
    return result;
}

// Scalar 1 in Montgomery form (for field arithmetic)
inline Fr make_fr_one_host() {
    Fr r;
    r.limbs[0] = FR_ONE_HOST[0];
    r.limbs[1] = FR_ONE_HOST[1];
    r.limbs[2] = FR_ONE_HOST[2];
    r.limbs[3] = FR_ONE_HOST[3];
    return r;
}

// Scalar 1 in standard integer form (for MSM - raw bit extraction)
// MSM expects scalars as plain integers, not Montgomery form
inline Fr make_fr_one_integer() {
    Fr r;
    r.limbs[0] = 1;
    r.limbs[1] = 0;
    r.limbs[2] = 0;
    r.limbs[3] = 0;
    return r;
}

// Create scalar from small integer (non-Montgomery form for MSM)
inline Fr make_fr_from_int(uint64_t val) {
    Fr r;
    r.limbs[0] = val;
    r.limbs[1] = 0;
    r.limbs[2] = 0;
    r.limbs[3] = 0;
    return r;
}

// Generate random scalar in standard integer form (for MSM)
inline Fr random_fr_integer(std::mt19937_64& rng) {
    uint256_t val;
    uint256_t modulus(FR_MODULUS_HOST);
    
    do {
        for (int i = 0; i < 4; i++) {
            val.limbs[i] = rng();
        }
        val.limbs[3] &= 0x7fffffffffffffffULL;
    } while (val >= modulus);
    
    Fr r;
    for (int i = 0; i < 4; i++) {
        r.limbs[i] = val.limbs[i];
    }
    return r;
}

inline Fr make_fr_zero_host() {
    Fr r;
    memset(r.limbs, 0, sizeof(r.limbs));
    return r;
}

} // namespace security_tests
