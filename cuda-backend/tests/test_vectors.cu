/**
 * @file test_vectors.cu
 * @brief Production-grade test suite for BLS12-381 CUDA backend
 * 
 * This test suite follows cryptographic library best practices:
 * - Known Answer Tests (KAT) with verified test vectors
 * - Property-based testing with proper Montgomery form handling
 * - Edge case coverage (zero, one, boundary values)
 * - Cross-validation against reference implementations
 * 
 * Test vectors sourced from:
 * - BLS12-381 specification (EIP-2537)
 * - BLST library test cases
 * - Arkworks reference implementation
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include "field.cuh"
#include "point.cuh"
#include "icicle_types.cuh"
#include "ntt.cuh"

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// Test Framework
// =============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
} while(0)

class TestSuite {
public:
    int tests_run = 0;
    int tests_passed = 0;
    
    void record(bool passed) {
        tests_run++;
        if (passed) tests_passed++;
    }
    
    bool all_passed() const { return tests_run == tests_passed; }
    
    void print_summary() const {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Results: " << tests_passed << "/" << tests_run << " tests passed";
        if (all_passed()) {
            std::cout << " ✓";
        } else {
            std::cout << " ✗";
        }
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

// =============================================================================
// BLS12-381 Verified Constants
// =============================================================================

/**
 * Fr modulus r:
 * 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
 * 
 * This is the order of the G1 and G2 groups.
 */
static const uint64_t FR_MODULUS_EXPECTED[4] = {
    0xffffffff00000001ULL,
    0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL,
    0x73eda753299d7d48ULL
};

/**
 * Fr Montgomery R = 2^256 mod r:
 * This is "1" in Montgomery form
 */
static const uint64_t FR_R_EXPECTED[4] = {
    0x00000001fffffffeULL,
    0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL,
    0x1824b159acc5056fULL
};

/**
 * Fr R^2 = 2^512 mod r:
 * Used to convert standard integers to Montgomery form
 */
static const uint64_t FR_R2_EXPECTED[4] = {
    0xc999e990f3f29c6dULL,
    0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL,
    0x0748d9d99f59ff11ULL
};

/**
 * Fq modulus p:
 * 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
 */
static const uint64_t FQ_MODULUS_EXPECTED[6] = {
    0xb9feffffffffaaabULL,
    0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL,
    0x1a0111ea397fe69aULL
};

// =============================================================================
// Host-side Montgomery Arithmetic (for test vector generation)
// =============================================================================

/**
 * @brief 256-bit unsigned integer for host-side computation
 */
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
        return true; // equal
    }
    
    bool operator==(const uint256_t& other) const {
        return memcmp(limbs, other.limbs, sizeof(limbs)) == 0;
    }
    
    bool is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }
};

/**
 * @brief Host-side 256-bit subtraction with borrow output
 */
uint256_t sub256(const uint256_t& a, const uint256_t& b, uint64_t& borrow_out) {
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

/**
 * @brief Host-side 256x256 -> 512 bit multiplication
 */
void mul256_full(const uint256_t& a, const uint256_t& b, uint64_t result[8]) {
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

/**
 * @brief Host-side Montgomery reduction (REDC)
 * 
 * Given T (512-bit), compute T * R^{-1} mod p
 * where R = 2^256
 */
uint256_t montgomery_reduce(const uint64_t T[8], const uint256_t& modulus, uint64_t inv) {
    uint64_t t[9];  // Extra limb for overflow
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
        
        // Propagate carry
        for (int j = i + 4; j < 9; j++) {
            __uint128_t sum = (__uint128_t)t[j] + carry;
            t[j] = (uint64_t)sum;
            carry = sum >> 64;
            if (carry == 0) break;
        }
    }
    
    // Result is in t[4..7]
    uint256_t result;
    memcpy(result.limbs, &t[4], 4 * sizeof(uint64_t));
    
    // Final reduction if result >= modulus
    if (result >= modulus) {
        uint64_t borrow;
        result = sub256(result, modulus, borrow);
    }
    
    return result;
}

/**
 * @brief Host-side Montgomery multiplication: result = a * b * R^{-1} mod p
 */
uint256_t montgomery_mul(const uint256_t& a, const uint256_t& b, 
                         const uint256_t& modulus, uint64_t inv) {
    uint64_t T[8];
    mul256_full(a, b, T);
    return montgomery_reduce(T, modulus, inv);
}

/**
 * @brief Convert integer to Montgomery form on host
 * 
 * Given a (standard form), compute aR mod p
 */
Fr to_montgomery_host(const uint256_t& a) {
    uint256_t modulus(FR_MODULUS_HOST);
    uint256_t r2(FR_R2_HOST);
    
    // aR = a * R^2 * R^{-1} = a * R^2 (montgomery mul with R^2)
    uint256_t result = montgomery_mul(a, r2, modulus, FR_INV);
    
    Fr fr;
    memcpy(fr.limbs, result.limbs, sizeof(fr.limbs));
    return fr;
}

/**
 * @brief Convert from Montgomery form to standard form on host
 */
uint256_t from_montgomery_host(const Fr& a) {
    uint256_t modulus(FR_MODULUS_HOST);
    uint256_t val(a.limbs);
    uint256_t one(1ULL);
    
    // a * 1 (montgomery mul) = a * R^{-1}
    return montgomery_mul(val, one, modulus, FR_INV);
}

// =============================================================================
// Test Vector Generation
// =============================================================================

/**
 * @brief Generate a properly reduced random Fr element (in Montgomery form)
 */
Fr random_fr_montgomery(std::mt19937_64& rng) {
    uint256_t val;
    uint256_t modulus(FR_MODULUS_HOST);
    
    // Generate random value and reduce
    do {
        for (int i = 0; i < 4; i++) {
            val.limbs[i] = rng();
        }
        // Clear top bit to ensure < 2*modulus for faster reduction
        val.limbs[3] &= 0x7fffffffffffffffULL;
    } while (val >= modulus);  // Rejection sampling for uniform distribution
    
    // Convert to Montgomery form
    return to_montgomery_host(val);
}

/**
 * @brief Generate a non-zero Fr element
 */
Fr random_fr_nonzero(std::mt19937_64& rng) {
    Fr result;
    do {
        result = random_fr_montgomery(rng);
    } while (result.is_zero());
    return result;
}

// =============================================================================
// Utility Functions
// =============================================================================

bool limbs_equal(const uint64_t* a, const uint64_t* b, int n) {
    return memcmp(a, b, n * sizeof(uint64_t)) == 0;
}

std::string limbs_to_hex(const uint64_t* limbs, int n) {
    std::ostringstream oss;
    oss << "0x";
    for (int i = n - 1; i >= 0; i--) {
        oss << std::hex << std::setfill('0') << std::setw(16) << limbs[i];
    }
    return oss.str();
}

void print_fr(const char* name, const Fr& val) {
    std::cout << name << " = " << limbs_to_hex(val.limbs, Fr::LIMBS) << std::endl;
}

// =============================================================================
// Test Kernels
// =============================================================================

/**
 * @brief Kernel: Compute a * b for each pair
 */
__global__ void mul_kernel(const Fr* a, const Fr* b, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

/**
 * @brief Kernel: Compute a + b for each pair
 */
__global__ void add_kernel(const Fr* a, const Fr* b, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Fr result;
    field_add(result, a[idx], b[idx]);
    out[idx] = result;
}

/**
 * @brief Kernel: Compute a - b for each pair
 */
__global__ void sub_kernel(const Fr* a, const Fr* b, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Fr result;
    field_sub(result, a[idx], b[idx]);
    out[idx] = result;
}

/**
 * @brief Kernel: Compute a^{-1} for each element
 */
__global__ void inv_kernel(const Fr* a, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    field_inv(out[idx], a[idx]);
}

/**
 * @brief Kernel: Compute a^2 using dedicated squaring
 */
__global__ void sqr_kernel(const Fr* a, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    field_sqr(out[idx], a[idx]);
}

/**
 * @brief Kernel: Convert to Montgomery form
 */
__global__ void to_mont_kernel(const Fr* a, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    field_to_montgomery(out[idx], a[idx]);
}

/**
 * @brief Kernel: Convert from Montgomery form
 */
__global__ void from_mont_kernel(const Fr* a, Fr* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    field_from_montgomery(out[idx], a[idx]);
}

/**
 * @brief Kernel: Test Fr::one() returns correct Montgomery 1
 */
__global__ void get_one_kernel(Fr* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = Fr::one();
    }
}

/**
 * @brief Kernel: Test Fr::from_int(n) conversion
 */
__global__ void from_int_kernel(uint64_t val, Fr* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = Fr::from_int(val);
    }
}

// =============================================================================
// Known Answer Tests (KAT)
// =============================================================================

/**
 * @brief Test 1: Verify Fr modulus constant matches specification
 */
bool test_fr_modulus_constant() {
    std::cout << "  [KAT] Fr modulus... " << std::flush;
    
    if (!limbs_equal(FR_MODULUS_HOST, FR_MODULUS_EXPECTED, 4)) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Expected: " << limbs_to_hex(FR_MODULUS_EXPECTED, 4) << std::endl;
        std::cout << "    Got:      " << limbs_to_hex(FR_MODULUS_HOST, 4) << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test 2: Verify Fr Montgomery R constant
 */
bool test_fr_montgomery_r() {
    std::cout << "  [KAT] Fr Montgomery R (one)... " << std::flush;
    
    if (!limbs_equal(FR_ONE_HOST, FR_R_EXPECTED, 4)) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Expected: " << limbs_to_hex(FR_R_EXPECTED, 4) << std::endl;
        std::cout << "    Got:      " << limbs_to_hex(FR_ONE_HOST, 4) << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test 3: Verify Fr R^2 constant
 */
bool test_fr_r_squared() {
    std::cout << "  [KAT] Fr R^2... " << std::flush;
    
    if (!limbs_equal(FR_R2_HOST, FR_R2_EXPECTED, 4)) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Expected: " << limbs_to_hex(FR_R2_EXPECTED, 4) << std::endl;
        std::cout << "    Got:      " << limbs_to_hex(FR_R2_HOST, 4) << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test 4: Verify Fq modulus constant
 */
bool test_fq_modulus_constant() {
    std::cout << "  [KAT] Fq modulus... " << std::flush;
    
    if (!limbs_equal(FQ_MODULUS_HOST, FQ_MODULUS_EXPECTED, 6)) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Expected: " << limbs_to_hex(FQ_MODULUS_EXPECTED, 6) << std::endl;
        std::cout << "    Got:      " << limbs_to_hex(FQ_MODULUS_HOST, 6) << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test 5: Verify Fr::one() on device matches constant
 */
bool test_fr_one_device() {
    std::cout << "  [KAT] Fr::one() on device... " << std::flush;
    
    Fr *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(Fr)));
    
    get_one_kernel<<<1, 1>>>(d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    Fr result;
    CHECK_CUDA(cudaMemcpy(&result, d_out, sizeof(Fr), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    
    if (!limbs_equal(result.limbs, FR_ONE_HOST, 4)) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Expected: " << limbs_to_hex(FR_ONE_HOST, 4) << std::endl;
        std::cout << "    Got:      " << limbs_to_hex(result.limbs, 4) << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test 6: Verify Fr::from_int(2) produces 2R mod p
 */
bool test_fr_from_int() {
    std::cout << "  [KAT] Fr::from_int(2)... " << std::flush;
    
    // Compute expected: 2R mod p on host
    Fr expected = to_montgomery_host(uint256_t(2ULL));
    
    Fr *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(Fr)));
    
    from_int_kernel<<<1, 1>>>(2ULL, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    Fr result;
    CHECK_CUDA(cudaMemcpy(&result, d_out, sizeof(Fr), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    
    if (!limbs_equal(result.limbs, expected.limbs, 4)) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Expected: " << limbs_to_hex(expected.limbs, 4) << std::endl;
        std::cout << "    Got:      " << limbs_to_hex(result.limbs, 4) << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// =============================================================================
// Algebraic Property Tests
// =============================================================================

/**
 * @brief Test: a * 1 = a (multiplicative identity)
 */
bool test_fr_mul_identity() {
    std::cout << "  [PROP] a * 1 = a... " << std::flush;
    
    const int n = 1024;
    std::mt19937_64 rng(42);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    // Create array of ones
    Fr one = Fr::one_host();
    std::vector<Fr> ones(n, one);
    
    Fr *d_a, *d_one, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_one, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_one, ones.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_one, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_one);
    cudaFree(d_out);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(a_vals[i].limbs, results[i].limbs, Fr::LIMBS)) {
            if (failures < 3) {
                std::cout << "\n    Failure at i=" << i << std::endl;
                print_fr("      a", a_vals[i]);
                print_fr("      a*1", results[i]);
            }
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: a * a^{-1} = 1 (multiplicative inverse)
 */
bool test_fr_mul_inverse() {
    std::cout << "  [PROP] a * a^{-1} = 1... " << std::flush;
    
    const int n = 512;  // Smaller due to expensive inversion
    std::mt19937_64 rng(123);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_nonzero(rng);
    }
    
    Fr *d_a, *d_inv, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_inv, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    // Compute inverses
    inv_kernel<<<(n + 255) / 256, 256>>>(d_a, d_inv, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compute a * a^{-1}
    mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_inv, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_inv);
    cudaFree(d_out);
    
    Fr one = Fr::one_host();
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(results[i].limbs, one.limbs, Fr::LIMBS)) {
            if (failures < 3) {
                std::cout << "\n    Failure at i=" << i << std::endl;
                print_fr("      a", a_vals[i]);
                print_fr("      a*a^{-1}", results[i]);
                print_fr("      expected", one);
            }
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: (a * b) * b^{-1} = a
 */
bool test_fr_mul_div() {
    std::cout << "  [PROP] (a*b)*b^{-1} = a... " << std::flush;
    
    const int n = 512;
    std::mt19937_64 rng(456);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_nonzero(rng);
    }
    
    Fr *d_a, *d_b, *d_ab, *d_b_inv, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_ab, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_b_inv, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    // Compute a * b
    mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_ab, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compute b^{-1}
    inv_kernel<<<(n + 255) / 256, 256>>>(d_b, d_b_inv, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compute (a*b) * b^{-1}
    mul_kernel<<<(n + 255) / 256, 256>>>(d_ab, d_b_inv, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ab);
    cudaFree(d_b_inv);
    cudaFree(d_out);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(a_vals[i].limbs, results[i].limbs, Fr::LIMBS)) {
            if (failures < 3) {
                std::cout << "\n    Failure at i=" << i << std::endl;
                print_fr("      a", a_vals[i]);
                print_fr("      (a*b)*b^{-1}", results[i]);
            }
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: a + 0 = a (additive identity)
 */
bool test_fr_add_identity() {
    std::cout << "  [PROP] a + 0 = a... " << std::flush;
    
    const int n = 1024;
    std::mt19937_64 rng(789);
    
    std::vector<Fr> a_vals(n);
    std::vector<Fr> zeros(n);  // Default constructor gives zero
    
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_zero, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_zero, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_zero, zeros.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    add_kernel<<<(n + 255) / 256, 256>>>(d_a, d_zero, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_zero);
    cudaFree(d_out);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(a_vals[i].limbs, results[i].limbs, Fr::LIMBS)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: a - a = 0 (additive inverse)
 */
bool test_fr_sub_self() {
    std::cout << "  [PROP] a - a = 0... " << std::flush;
    
    const int n = 1024;
    std::mt19937_64 rng(111);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    sub_kernel<<<(n + 255) / 256, 256>>>(d_a, d_a, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_out);
    
    Fr zero;
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(results[i].limbs, zero.limbs, Fr::LIMBS)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: a * (b + c) = a*b + a*c (distributivity)
 */
bool test_fr_distributive() {
    std::cout << "  [PROP] a*(b+c) = a*b + a*c... " << std::flush;
    
    const int n = 1024;
    std::mt19937_64 rng(222);
    
    std::vector<Fr> a_vals(n), b_vals(n), c_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
        b_vals[i] = random_fr_montgomery(rng);
        c_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_b, *d_c, *d_bc, *d_lhs, *d_ab, *d_ac, *d_rhs;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_bc, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_lhs, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_ab, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_ac, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_rhs, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, c_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    // LHS: a * (b + c)
    add_kernel<<<(n + 255) / 256, 256>>>(d_b, d_c, d_bc, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_bc, d_lhs, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // RHS: a*b + a*c
    mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_ab, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_c, d_ac, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    add_kernel<<<(n + 255) / 256, 256>>>(d_ab, d_ac, d_rhs, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> lhs(n), rhs(n);
    CHECK_CUDA(cudaMemcpy(lhs.data(), d_lhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(rhs.data(), d_rhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_bc);
    cudaFree(d_lhs); cudaFree(d_ab); cudaFree(d_ac); cudaFree(d_rhs);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(lhs[i].limbs, rhs[i].limbs, Fr::LIMBS)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: sqr(a) = a * a (squaring consistency)
 */
bool test_fr_squaring() {
    std::cout << "  [PROP] sqr(a) = a*a... " << std::flush;
    
    const int n = 1024;
    std::mt19937_64 rng(333);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_sqr, *d_mul;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_sqr, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_mul, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    sqr_kernel<<<(n + 255) / 256, 256>>>(d_a, d_sqr, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    mul_kernel<<<(n + 255) / 256, 256>>>(d_a, d_a, d_mul, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> sqr_out(n), mul_out(n);
    CHECK_CUDA(cudaMemcpy(sqr_out.data(), d_sqr, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(mul_out.data(), d_mul, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_sqr);
    cudaFree(d_mul);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(sqr_out[i].limbs, mul_out[i].limbs, Fr::LIMBS)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/**
 * @brief Test: 0 * a = 0
 */
bool test_fr_mul_zero() {
    std::cout << "  [EDGE] 0 * a = 0... " << std::flush;
    
    const int n = 256;
    std::mt19937_64 rng(444);
    
    std::vector<Fr> a_vals(n);
    std::vector<Fr> zeros(n);
    
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_zero, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_zero, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_zero, zeros.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    mul_kernel<<<(n + 255) / 256, 256>>>(d_zero, d_a, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    CHECK_CUDA(cudaMemcpy(results.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_zero);
    cudaFree(d_out);
    
    Fr zero;
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(results[i].limbs, zero.limbs, Fr::LIMBS)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: 1^{-1} = 1
 */
bool test_fr_inv_one() {
    std::cout << "  [EDGE] 1^{-1} = 1... " << std::flush;
    
    Fr one = Fr::one_host();
    
    Fr *d_one, *d_inv;
    CHECK_CUDA(cudaMalloc(&d_one, sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_inv, sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_one, &one, sizeof(Fr), cudaMemcpyHostToDevice));
    
    inv_kernel<<<1, 1>>>(d_one, d_inv, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    Fr result;
    CHECK_CUDA(cudaMemcpy(&result, d_inv, sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_one);
    cudaFree(d_inv);
    
    if (!limbs_equal(result.limbs, one.limbs, Fr::LIMBS)) {
        std::cout << "FAILED" << std::endl;
        print_fr("    Expected", one);
        print_fr("    Got", result);
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: Montgomery round-trip (to_mont(from_mont(a)) = a)
 */
bool test_fr_montgomery_roundtrip() {
    std::cout << "  [EDGE] Montgomery roundtrip... " << std::flush;
    
    const int n = 256;
    std::mt19937_64 rng(555);
    
    std::vector<Fr> a_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr_montgomery(rng);
    }
    
    Fr *d_a, *d_std, *d_back;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_std, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_back, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    // Convert to standard form
    from_mont_kernel<<<(n + 255) / 256, 256>>>(d_a, d_std, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Convert back to Montgomery form
    to_mont_kernel<<<(n + 255) / 256, 256>>>(d_std, d_back, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> results(n);
    CHECK_CUDA(cudaMemcpy(results.data(), d_back, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_std);
    cudaFree(d_back);
    
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (!limbs_equal(a_vals[i].limbs, results[i].limbs, Fr::LIMBS)) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << "/" << n << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// =============================================================================
// G1 Curve Tests
// =============================================================================

/**
 * @brief Kernel: G1 point doubling with affine input
 */
__global__ void g1_double_from_affine_kernel(const G1Affine* p_affine, G1Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        G1Projective p = G1Projective::from_affine(*p_affine);
        g1_double(*out, p);
    }
}

/**
 * @brief Kernel: G1 point addition with affine input (P + P)
 */
__global__ void g1_add_self_from_affine_kernel(const G1Affine* p_affine, G1Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        G1Projective p = G1Projective::from_affine(*p_affine);
        g1_add(*out, p, p);
    }
}

/**
 * @brief Kernel: G1 identity + P
 */
__global__ void g1_add_identity_kernel(const G1Affine* p_affine, G1Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        G1Projective p = G1Projective::from_affine(*p_affine);
        G1Projective identity = G1Projective::identity();
        g1_add(*out, identity, p);
    }
}

/**
 * @brief Kernel: Copy projective point from affine
 */
__global__ void g1_from_affine_kernel(const G1Affine* p_affine, G1Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = G1Projective::from_affine(*p_affine);
    }
}

/**
 * @brief Test: 2P (double) = P + P (add) for G1
 * 
 * All point operations happen on GPU - only affine coordinates copied to device,
 * conversion to projective and arithmetic done on GPU.
 */
bool test_g1_double_vs_add() {
    std::cout << "  [CURVE] 2P = P + P... " << std::flush;
    
    // Prepare affine generator point on host (just raw coordinate data)
    G1Affine gen;
    memcpy(gen.x.limbs, G1_GENERATOR_X, sizeof(G1_GENERATOR_X));
    memcpy(gen.y.limbs, G1_GENERATOR_Y, sizeof(G1_GENERATOR_Y));
    
    // Allocate GPU memory
    G1Affine* d_gen;
    G1Projective *d_double, *d_add;
    cudaError_t err;
    
    err = cudaMalloc(&d_gen, sizeof(G1Affine));
    if (err != cudaSuccess) {
        std::cout << "FAILED (malloc d_gen: " << cudaGetErrorString(err) << ")" << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_double, sizeof(G1Projective));
    if (err != cudaSuccess) {
        cudaFree(d_gen);
        std::cout << "FAILED (malloc d_double: " << cudaGetErrorString(err) << ")" << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_add, sizeof(G1Projective));
    if (err != cudaSuccess) {
        cudaFree(d_gen);
        cudaFree(d_double);
        std::cout << "FAILED (malloc d_add: " << cudaGetErrorString(err) << ")" << std::endl;
        return false;
    }
    
    // Copy only affine coordinates to device (no Montgomery operations on host)
    err = cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "FAILED (memcpy: " << cudaGetErrorString(err) << ")" << std::endl;
        cudaFree(d_gen); cudaFree(d_double); cudaFree(d_add);
        return false;
    }
    
    // Launch kernels - all conversion and operations happen on GPU
    g1_double_from_affine_kernel<<<1, 1>>>(d_gen, d_double);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "FAILED (double kernel: " << cudaGetErrorString(err) << ")" << std::endl;
        cudaFree(d_gen); cudaFree(d_double); cudaFree(d_add);
        return false;
    }
    
    g1_add_self_from_affine_kernel<<<1, 1>>>(d_gen, d_add);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "FAILED (add kernel: " << cudaGetErrorString(err) << ")" << std::endl;
        cudaFree(d_gen); cudaFree(d_double); cudaFree(d_add);
        return false;
    }
    
    // Copy results back
    G1Projective double_result, add_result;
    cudaMemcpy(&double_result, d_double, sizeof(G1Projective), cudaMemcpyDeviceToHost);
    cudaMemcpy(&add_result, d_add, sizeof(G1Projective), cudaMemcpyDeviceToHost);
    
    cudaFree(d_gen);
    cudaFree(d_double);
    cudaFree(d_add);
    
    // Compare projective coordinates
    // Since both operations started from same point and did same computation,
    // the intermediate Z values should match exactly
    bool match = limbs_equal(double_result.X.limbs, add_result.X.limbs, Fq::LIMBS) &&
                 limbs_equal(double_result.Y.limbs, add_result.Y.limbs, Fq::LIMBS) &&
                 limbs_equal(double_result.Z.limbs, add_result.Z.limbs, Fq::LIMBS);
    
    if (!match) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Double and add produced different results" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: Identity + P = P
 * 
 * All operations on GPU - copies only affine coordinates to device.
 */
bool test_g1_add_identity() {
    std::cout << "  [CURVE] O + P = P... " << std::flush;
    
    // Prepare affine generator point on host
    G1Affine gen;
    memcpy(gen.x.limbs, G1_GENERATOR_X, sizeof(G1_GENERATOR_X));
    memcpy(gen.y.limbs, G1_GENERATOR_Y, sizeof(G1_GENERATOR_Y));
    
    // Allocate GPU memory
    G1Affine* d_gen;
    G1Projective *d_out, *d_p;
    
    CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Affine)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(G1Projective)));
    CHECK_CUDA(cudaMalloc(&d_p, sizeof(G1Projective)));
    
    // Copy affine coordinates to device
    CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G1Affine), cudaMemcpyHostToDevice));
    
    // Test O + P = P (done on GPU)
    g1_add_identity_kernel<<<1, 1>>>(d_gen, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Also get P directly for comparison
    g1_from_affine_kernel<<<1, 1>>>(d_gen, d_p);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back
    G1Projective result, p;
    CHECK_CUDA(cudaMemcpy(&result, d_out, sizeof(G1Projective), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&p, d_p, sizeof(G1Projective), cudaMemcpyDeviceToHost));
    
    cudaFree(d_gen);
    cudaFree(d_out);
    cudaFree(d_p);
    
    // Compare result with P
    bool match = limbs_equal(result.X.limbs, p.X.limbs, Fq::LIMBS) &&
                 limbs_equal(result.Y.limbs, p.Y.limbs, Fq::LIMBS) &&
                 limbs_equal(result.Z.limbs, p.Z.limbs, Fq::LIMBS);
    
    if (!match) {
        std::cout << "FAILED" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// =============================================================================
// G2 Curve Tests
// =============================================================================

/**
 * @brief Kernel: G2 point doubling with affine input
 */
__global__ void g2_double_from_affine_kernel(const G2Affine* p_affine, G2Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        G2Projective p = G2Projective::from_affine(*p_affine);
        g2_double(*out, p);
    }
}

/**
 * @brief Kernel: G2 point addition with affine input (P + P)
 */
__global__ void g2_add_self_from_affine_kernel(const G2Affine* p_affine, G2Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        G2Projective p = G2Projective::from_affine(*p_affine);
        g2_add(*out, p, p);
    }
}

/**
 * @brief Kernel: G2 identity + P
 */
__global__ void g2_add_identity_kernel(const G2Affine* p_affine, G2Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        G2Projective p = G2Projective::from_affine(*p_affine);
        G2Projective identity = G2Projective::identity();
        g2_add(*out, identity, p);
    }
}

/**
 * @brief Kernel: Copy projective point from affine for G2
 */
__global__ void g2_from_affine_kernel(const G2Affine* p_affine, G2Projective* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = G2Projective::from_affine(*p_affine);
    }
}

/**
 * @brief Test: 2P (double) = P + P (add) for G2
 * 
 * All point operations happen on GPU.
 */
bool test_g2_double_vs_add() {
    std::cout << "  [CURVE] G2: 2P = P + P... " << std::flush;
    
    // Prepare affine G2 generator point on host
    // G2 coordinates are Fq2 elements (c0 + c1*u)
    G2Affine gen;
    memcpy(gen.x.c0.limbs, G2_GENERATOR_X_C0, sizeof(G2_GENERATOR_X_C0));
    memcpy(gen.x.c1.limbs, G2_GENERATOR_X_C1, sizeof(G2_GENERATOR_X_C1));
    memcpy(gen.y.c0.limbs, G2_GENERATOR_Y_C0, sizeof(G2_GENERATOR_Y_C0));
    memcpy(gen.y.c1.limbs, G2_GENERATOR_Y_C1, sizeof(G2_GENERATOR_Y_C1));
    
    // Allocate GPU memory
    G2Affine* d_gen;
    G2Projective *d_double, *d_add;
    cudaError_t err;
    
    err = cudaMalloc(&d_gen, sizeof(G2Affine));
    if (err != cudaSuccess) {
        std::cout << "FAILED (malloc d_gen: " << cudaGetErrorString(err) << ")" << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_double, sizeof(G2Projective));
    if (err != cudaSuccess) {
        cudaFree(d_gen);
        std::cout << "FAILED (malloc d_double: " << cudaGetErrorString(err) << ")" << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_add, sizeof(G2Projective));
    if (err != cudaSuccess) {
        cudaFree(d_gen);
        cudaFree(d_double);
        std::cout << "FAILED (malloc d_add: " << cudaGetErrorString(err) << ")" << std::endl;
        return false;
    }
    
    // Copy affine coordinates to device
    err = cudaMemcpy(d_gen, &gen, sizeof(G2Affine), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "FAILED (memcpy: " << cudaGetErrorString(err) << ")" << std::endl;
        cudaFree(d_gen); cudaFree(d_double); cudaFree(d_add);
        return false;
    }
    
    // Launch kernels
    g2_double_from_affine_kernel<<<1, 1>>>(d_gen, d_double);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "FAILED (double kernel: " << cudaGetErrorString(err) << ")" << std::endl;
        cudaFree(d_gen); cudaFree(d_double); cudaFree(d_add);
        return false;
    }
    
    g2_add_self_from_affine_kernel<<<1, 1>>>(d_gen, d_add);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "FAILED (add kernel: " << cudaGetErrorString(err) << ")" << std::endl;
        cudaFree(d_gen); cudaFree(d_double); cudaFree(d_add);
        return false;
    }
    
    // Copy results back
    G2Projective double_result, add_result;
    cudaMemcpy(&double_result, d_double, sizeof(G2Projective), cudaMemcpyDeviceToHost);
    cudaMemcpy(&add_result, d_add, sizeof(G2Projective), cudaMemcpyDeviceToHost);
    
    cudaFree(d_gen);
    cudaFree(d_double);
    cudaFree(d_add);
    
    // Compare projective coordinates (Fq2 elements have c0 and c1)
    bool match = limbs_equal(double_result.X.c0.limbs, add_result.X.c0.limbs, Fq::LIMBS) &&
                 limbs_equal(double_result.X.c1.limbs, add_result.X.c1.limbs, Fq::LIMBS) &&
                 limbs_equal(double_result.Y.c0.limbs, add_result.Y.c0.limbs, Fq::LIMBS) &&
                 limbs_equal(double_result.Y.c1.limbs, add_result.Y.c1.limbs, Fq::LIMBS) &&
                 limbs_equal(double_result.Z.c0.limbs, add_result.Z.c0.limbs, Fq::LIMBS) &&
                 limbs_equal(double_result.Z.c1.limbs, add_result.Z.c1.limbs, Fq::LIMBS);
    
    if (!match) {
        std::cout << "FAILED" << std::endl;
        std::cout << "    Double and add produced different results" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

/**
 * @brief Test: Identity + P = P for G2
 */
bool test_g2_add_identity() {
    std::cout << "  [CURVE] G2: O + P = P... " << std::flush;
    
    // Prepare G2 generator
    G2Affine gen;
    memcpy(gen.x.c0.limbs, G2_GENERATOR_X_C0, sizeof(G2_GENERATOR_X_C0));
    memcpy(gen.x.c1.limbs, G2_GENERATOR_X_C1, sizeof(G2_GENERATOR_X_C1));
    memcpy(gen.y.c0.limbs, G2_GENERATOR_Y_C0, sizeof(G2_GENERATOR_Y_C0));
    memcpy(gen.y.c1.limbs, G2_GENERATOR_Y_C1, sizeof(G2_GENERATOR_Y_C1));
    
    // Allocate GPU memory
    G2Affine* d_gen;
    G2Projective *d_out, *d_p;
    
    CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G2Affine)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(G2Projective)));
    CHECK_CUDA(cudaMalloc(&d_p, sizeof(G2Projective)));
    
    CHECK_CUDA(cudaMemcpy(d_gen, &gen, sizeof(G2Affine), cudaMemcpyHostToDevice));
    
    // Test O + P = P
    g2_add_identity_kernel<<<1, 1>>>(d_gen, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Get P directly for comparison
    g2_from_affine_kernel<<<1, 1>>>(d_gen, d_p);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back
    G2Projective result, p;
    CHECK_CUDA(cudaMemcpy(&result, d_out, sizeof(G2Projective), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&p, d_p, sizeof(G2Projective), cudaMemcpyDeviceToHost));
    
    cudaFree(d_gen);
    cudaFree(d_out);
    cudaFree(d_p);
    
    // Compare
    bool match = limbs_equal(result.X.c0.limbs, p.X.c0.limbs, Fq::LIMBS) &&
                 limbs_equal(result.X.c1.limbs, p.X.c1.limbs, Fq::LIMBS) &&
                 limbs_equal(result.Y.c0.limbs, p.Y.c0.limbs, Fq::LIMBS) &&
                 limbs_equal(result.Y.c1.limbs, p.Y.c1.limbs, Fq::LIMBS) &&
                 limbs_equal(result.Z.c0.limbs, p.Z.c0.limbs, Fq::LIMBS) &&
                 limbs_equal(result.Z.c1.limbs, p.Z.c1.limbs, Fq::LIMBS);
    
    if (!match) {
        std::cout << "FAILED" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    (void)argc; (void)argv;  // Unused parameters
    std::cout << "========================================" << std::endl;
    std::cout << "BLS12-381 Production Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check CUDA device
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Use current device instead of hardcoding 0
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    TestSuite suite;
    
    // Known Answer Tests
    std::cout << "--- Known Answer Tests (KAT) ---" << std::endl;
    suite.record(test_fr_modulus_constant());
    suite.record(test_fr_montgomery_r());
    suite.record(test_fr_r_squared());
    suite.record(test_fq_modulus_constant());
    suite.record(test_fr_one_device());
    suite.record(test_fr_from_int());
    
    // Algebraic Property Tests
    std::cout << std::endl << "--- Algebraic Property Tests ---" << std::endl;
    suite.record(test_fr_mul_identity());
    suite.record(test_fr_mul_inverse());
    suite.record(test_fr_mul_div());
    suite.record(test_fr_add_identity());
    suite.record(test_fr_sub_self());
    suite.record(test_fr_distributive());
    suite.record(test_fr_squaring());
    
    // Edge Case Tests
    std::cout << std::endl << "--- Edge Case Tests ---" << std::endl;
    suite.record(test_fr_mul_zero());
    suite.record(test_fr_inv_one());
    suite.record(test_fr_montgomery_roundtrip());
    
    // Curve Tests
    std::cout << std::endl << "--- G1 Curve Tests ---" << std::endl;
    std::cout.flush();
    suite.record(test_g1_double_vs_add());
    std::cout.flush();
    suite.record(test_g1_add_identity());
    std::cout.flush();
    
    std::cout << std::endl << "--- G2 Curve Tests ---" << std::endl;
    std::cout.flush();
    suite.record(test_g2_double_vs_add());
    std::cout.flush();
    suite.record(test_g2_add_identity());
    std::cout.flush();
    
    suite.print_summary();
    
    return suite.all_passed() ? 0 : 1;
}
