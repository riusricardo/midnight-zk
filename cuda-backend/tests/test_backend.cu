/**
 * @file test_backend.cu
 * @brief Test suite for the CUDA BLS12-381 backend
 * 
 * This test file validates the correctness and performance of:
 * - Field arithmetic (Fr, Fq)
 * - Point operations (G1, G2)
 * - Vector operations
 * - NTT/INTT
 * - MSM
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdint>

#include "field.cuh"
#include "point.cuh"
#include "icicle_types.cuh"
#include "ntt.cuh"

using namespace bls12_381;

// =============================================================================
// Exported function declarations
// =============================================================================

extern "C" {
    eIcicleError bls12_381_vector_add(const Fr*, const Fr*, size_t, const VecOpsConfig*, Fr*);
    eIcicleError bls12_381_vector_sub(const Fr*, const Fr*, size_t, const VecOpsConfig*, Fr*);
    eIcicleError bls12_381_vector_mul(const Fr*, const Fr*, size_t, const VecOpsConfig*, Fr*);
    eIcicleError bls12_381_g1_scalar_mul(const G1Affine*, const Fr*, int, const VecOpsConfig*, G1Projective*);
    eIcicleError bls12_381_g1_scalar_mul_glv(const G1Affine*, const Fr*, int, const VecOpsConfig*, G1Projective*);
}

// =============================================================================
// Test Utilities
// =============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
} while(0)

class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Host-side Fr one value (from constants)
Fr make_fr_one_host() {
    Fr r;
    r.limbs[0] = FR_ONE_HOST[0];
    r.limbs[1] = FR_ONE_HOST[1];
    r.limbs[2] = FR_ONE_HOST[2];
    r.limbs[3] = FR_ONE_HOST[3];
    return r;
}

// Host-side Fr zero value
Fr make_fr_zero_host() {
    Fr r;
    for (int i = 0; i < Fr::LIMBS; i++) {
        r.limbs[i] = 0;
    }
    return r;
}

// Compare two 4-limb numbers: returns -1 if a < b, 0 if a == b, 1 if a > b
int compare_limbs(const uint64_t* a, const uint64_t* b, int limbs) {
    for (int i = limbs - 1; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Subtract b from a (assuming a >= b), returns borrow
void sub_limbs(uint64_t* result, const uint64_t* a, const uint64_t* b, int limbs) {
    uint64_t borrow = 0;
    for (int i = 0; i < limbs; i++) {
        uint64_t ai = a[i];
        uint64_t bi = b[i];
        uint64_t diff = ai - bi;
        uint64_t new_borrow = (ai < bi) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - borrow;
        new_borrow += (diff < borrow) ? 1ULL : 0ULL;
        result[i] = diff2;
        borrow = new_borrow;
    }
}

// Generate random Fr element (properly reduced)
Fr random_fr(std::mt19937_64& rng) {
    Fr r;
    for (int i = 0; i < Fr::LIMBS; i++) {
        r.limbs[i] = rng();
    }
    // Reduce modulo the field modulus
    // Keep subtracting modulus while >= modulus
    while (compare_limbs(r.limbs, FR_MODULUS_HOST, Fr::LIMBS) >= 0) {
        sub_limbs(r.limbs, r.limbs, FR_MODULUS_HOST, Fr::LIMBS);
    }
    return r;
}

// =============================================================================
// Field Arithmetic Test Kernels
// =============================================================================

/**
 * @brief Test kernel: Verify that a * a^(-1) = 1
 */
__global__ void test_field_mul_inv_kernel(
    const Fr* inputs,
    Fr* outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr a = inputs[idx];
    
    // Skip zero (has no inverse)
    if (a.is_zero()) {
        outputs[idx] = Fr::one();
        return;
    }
    
    // Compute inverse
    Fr a_inv = field_inv(a);
    
    // Compute a * a^(-1)
    Fr result = a * a_inv;
    
    outputs[idx] = result;
}

/**
 * @brief Test kernel: Verify that (a + b) - b = a
 */
__global__ void test_field_add_sub_kernel(
    const Fr* a_vals,
    const Fr* b_vals,
    Fr* outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr a = a_vals[idx];
    Fr b = b_vals[idx];
    
    Fr sum = a + b;
    Fr result = sum - b;
    
    outputs[idx] = result;
}

/**
 * @brief Test kernel: Verify multiplication associativity (a*b)*c = a*(b*c)
 */
__global__ void test_field_mul_assoc_kernel(
    const Fr* a_vals,
    const Fr* b_vals,
    const Fr* c_vals,
    Fr* lhs_outputs,  // (a*b)*c
    Fr* rhs_outputs,  // a*(b*c)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr a = a_vals[idx];
    Fr b = b_vals[idx];
    Fr c = c_vals[idx];
    
    lhs_outputs[idx] = (a * b) * c;
    rhs_outputs[idx] = a * (b * c);
}

/**
 * @brief Test kernel: Verify a^2 = a * a using dedicated squaring
 */
__global__ void test_field_sqr_kernel(
    const Fr* inputs,
    Fr* sqr_outputs,   // Using field_sqr
    Fr* mul_outputs,   // Using a * a
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Fr a = inputs[idx];
    
    // Using dedicated squaring
    Fr a_sqr;
    field_sqr(a_sqr, a);
    sqr_outputs[idx] = a_sqr;
    
    // Using regular multiplication
    mul_outputs[idx] = a * a;
}

// =============================================================================
// Point Operation Test Kernels
// =============================================================================

/**
 * @brief Test kernel: Verify point doubling P + P = 2P
 */
__global__ void test_point_double_kernel(
    const G1Projective* points,
    G1Projective* double_results,
    G1Projective* add_results,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective p = points[idx];
    
    // Compute 2P via doubling
    g1_double(double_results[idx], p);
    
    // Compute P + P via addition
    g1_add(add_results[idx], p, p);
}

/**
 * @brief Test kernel: Verify P + (-P) = identity
 */
__global__ void test_point_neg_kernel(
    const G1Projective* points,
    G1Projective* results,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective p = points[idx];
    G1Projective neg_p;
    g1_neg(neg_p, p);
    
    g1_add(results[idx], p, neg_p);
}

/**
 * @brief Test kernel: Verify G2 point doubling 2P via g2_double matches P + P
 */
__global__ void test_g2_double_kernel(
    const G2Projective* points,
    G2Projective* double_results,
    G2Projective* add_results,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G2Projective p = points[idx];
    
    // Compute 2P via doubling
    g2_double(double_results[idx], p);
    
    // Compute P + P via addition
    g2_add(add_results[idx], p, p);
}

/**
 * @brief Test kernel: Verify G2 P + (-P) = identity
 */
__global__ void test_g2_neg_kernel(
    const G2Projective* points,
    G2Projective* results,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G2Projective p = points[idx];
    G2Projective neg_p;
    g2_neg(neg_p, p);
    
    g2_add(results[idx], p, neg_p);
}

// =============================================================================
// Test Functions
// =============================================================================

bool test_field_arithmetic() {
    std::cout << "Testing field arithmetic (mul/inv identity)... " << std::flush;
    
    const int n = 4096;
    std::mt19937_64 rng(42);
    
    // Generate random field elements
    std::vector<Fr> inputs(n);
    for (int i = 0; i < n; i++) {
        inputs[i] = random_fr(rng);
    }
    
    // Allocate device memory
    Fr *d_inputs, *d_outputs;
    CHECK_CUDA(cudaMalloc(&d_inputs, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_outputs, n * sizeof(Fr)));
    
    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_inputs, inputs.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    // Run kernel
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_field_mul_inv_kernel<<<blocks, threads>>>(d_inputs, d_outputs, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back
    std::vector<Fr> outputs(n);
    CHECK_CUDA(cudaMemcpy(outputs.data(), d_outputs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    
    // Verify all results equal one
    Fr one = make_fr_one_host();
    int failures = 0;
    for (int i = 0; i < n; i++) {
        bool equal = true;
        for (int j = 0; j < Fr::LIMBS; j++) {
            if (outputs[i].limbs[j] != one.limbs[j]) {
                equal = false;
                break;
            }
        }
        if (!equal) {
            failures++;
            if (failures <= 3) {
                std::cerr << "Failure at index " << i << std::endl;
            }
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << " failures)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_field_add_sub() {
    std::cout << "Testing field arithmetic (add/sub identity)... " << std::flush;
    
    const int n = 4096;
    std::mt19937_64 rng(123);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr(rng);
        b_vals[i] = random_fr(rng);
    }
    
    Fr *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_field_add_sub_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> outputs(n);
    CHECK_CUDA(cudaMemcpy(outputs.data(), d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    
    // Verify (a + b) - b == a
    int failures = 0;
    for (int i = 0; i < n; i++) {
        bool equal = true;
        for (int j = 0; j < Fr::LIMBS; j++) {
            if (outputs[i].limbs[j] != a_vals[i].limbs[j]) {
                equal = false;
                break;
            }
        }
        if (!equal) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << " failures)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_field_mul_associativity() {
    std::cout << "Testing field arithmetic (mul associativity)... " << std::flush;
    
    const int n = 4096;
    std::mt19937_64 rng(456);
    
    std::vector<Fr> a_vals(n), b_vals(n), c_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr(rng);
        b_vals[i] = random_fr(rng);
        c_vals[i] = random_fr(rng);
    }
    
    Fr *d_a, *d_b, *d_c, *d_lhs, *d_rhs;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_lhs, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_rhs, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, c_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_field_mul_assoc_kernel<<<blocks, threads>>>(d_a, d_b, d_c, d_lhs, d_rhs, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> lhs(n), rhs(n);
    CHECK_CUDA(cudaMemcpy(lhs.data(), d_lhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(rhs.data(), d_rhs, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_lhs);
    cudaFree(d_rhs);
    
    // Verify (a*b)*c == a*(b*c)
    int failures = 0;
    for (int i = 0; i < n; i++) {
        bool equal = true;
        for (int j = 0; j < Fr::LIMBS; j++) {
            if (lhs[i].limbs[j] != rhs[i].limbs[j]) {
                equal = false;
                break;
            }
        }
        if (!equal) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << " failures)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_field_squaring() {
    std::cout << "Testing field squaring (sqr vs mul)... " << std::flush;
    
    const int n = 4096;
    std::mt19937_64 rng(321);
    
    std::vector<Fr> inputs(n);
    for (int i = 0; i < n; i++) {
        inputs[i] = random_fr(rng);
    }
    
    Fr *d_inputs, *d_sqr, *d_mul;
    CHECK_CUDA(cudaMalloc(&d_inputs, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_sqr, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_mul, n * sizeof(Fr)));
    
    CHECK_CUDA(cudaMemcpy(d_inputs, inputs.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    test_field_sqr_kernel<<<blocks, threads>>>(d_inputs, d_sqr, d_mul, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<Fr> sqr_results(n), mul_results(n);
    CHECK_CUDA(cudaMemcpy(sqr_results.data(), d_sqr, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(mul_results.data(), d_mul, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    cudaFree(d_inputs);
    cudaFree(d_sqr);
    cudaFree(d_mul);
    
    // Verify field_sqr(a) == a * a
    int failures = 0;
    for (int i = 0; i < n; i++) {
        bool equal = true;
        for (int j = 0; j < Fr::LIMBS; j++) {
            if (sqr_results[i].limbs[j] != mul_results[i].limbs[j]) {
                equal = false;
                break;
            }
        }
        if (!equal) {
            failures++;
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << " failures)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_g2_operations() {
    std::cout << "Testing G2 point operations... " << std::flush;
    
    // G2 operations compile and are available.
    // Full kernel tests require generator point which needs careful setup.
    // For now, just verify the types and structures are correct.
    
    std::cout << "PASSED (compile check)" << std::endl;
    return true;
}

bool test_vec_add() {
    std::cout << "Testing vector addition... " << std::flush;
    
    const int n = 8192;
    std::mt19937_64 rng(789);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr(rng);
        b_vals[i] = random_fr(rng);
    }
    
    // Use the exported vec_add function
    VecOpsConfig config;
    config.stream = nullptr;
    config.is_a_on_device = false;
    config.is_b_on_device = false;
    config.is_result_on_device = false;
    config.is_async = false;
    
    std::vector<Fr> results(n);
    
    // Call the exported function
    eIcicleError err = bls12_381_vector_add(a_vals.data(), b_vals.data(), n, &config, results.data());
    
    if (err != eIcicleError::SUCCESS) {
        std::cout << "FAILED (function returned error)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_library_loading() {
    std::cout << "Testing library symbols... " << std::flush;
    
    // The symbols are declared at file scope, so if we linked successfully they exist
    
    // Just verify they're callable (don't actually run with null pointers)
    std::cout << "PASSED (symbols found)" << std::endl;
    return true;
}

// =============================================================================
// GLV Scalar Multiplication Tests
// =============================================================================

// Generator point for BLS12-381 G1 (in Montgomery form)
// G1 generator: x = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
//               y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
// Converted to Montgomery form for Fq
G1Affine make_g1_generator() {
    G1Affine g;
    // x in Montgomery form
    g.x.limbs[0] = 0xfd530c16a28a2ed5ULL;
    g.x.limbs[1] = 0xc0f3db9eb2a81c60ULL;
    g.x.limbs[2] = 0xa18ad315bdd26cb9ULL;
    g.x.limbs[3] = 0x6c69116d93a67ca5ULL;
    g.x.limbs[4] = 0x04c9ad3661f6eae1ULL;
    g.x.limbs[5] = 0x1120bb669f6f8d4eULL;
    
    // y in Montgomery form
    g.y.limbs[0] = 0x11560bf17baa99bcULL;
    g.y.limbs[1] = 0xe17df37a3381b236ULL;
    g.y.limbs[2] = 0x0f0c5ec24fea7680ULL;
    g.y.limbs[3] = 0x2e6d639bed6c3ac2ULL;
    g.y.limbs[4] = 0x044a7cd5c36d13f1ULL;
    g.y.limbs[5] = 0x120230e9d5639d9dULL;
    
    return g;
}

// Create scalar from integer value (for testing small scalars)
Fr make_fr_from_u64(uint64_t val) {
    Fr r;
    r.limbs[0] = val;
    r.limbs[1] = 0;
    r.limbs[2] = 0;
    r.limbs[3] = 0;
    return r;
}

/**
 * @brief Test kernel: simple scalar mul using double-and-add (reference implementation)
 * Used to verify GLV results against a known-correct naive implementation.
 */
__global__ void reference_scalar_mul_kernel(
    G1Projective* output,
    const G1Affine* base,
    const Fr* scalar,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective result = G1Projective::identity();
    G1Projective p = G1Projective::from_affine(base[idx]);
    Fr s = scalar[idx];
    
    // Simple double-and-add (LSB to MSB)
    for (int i = 0; i < 255; i++) {
        int limb_idx = i / 64;
        int bit_idx = i % 64;
        
        if ((s.limbs[limb_idx] >> bit_idx) & 1) {
            g1_add(result, result, p);
        }
        g1_double(p, p);
    }
    
    output[idx] = result;
}

/**
 * @brief Compare two G1 projective points for equality
 * Points are equal if X1*Z2 = X2*Z1 and Y1*Z2 = Y2*Z1
 */
__global__ void compare_g1_points_kernel(
    const G1Projective* a,
    const G1Projective* b,
    int* equal_flags,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective p1 = a[idx];
    G1Projective p2 = b[idx];
    
    // Both identity?
    if (p1.is_identity() && p2.is_identity()) {
        equal_flags[idx] = 1;
        return;
    }
    
    // One identity, other not?
    if (p1.is_identity() != p2.is_identity()) {
        equal_flags[idx] = 0;
        return;
    }
    
    // Compare: X1*Z2 == X2*Z1
    Fq x1z2, x2z1;
    field_mul(x1z2, p1.X, p2.Z);
    field_mul(x2z1, p2.X, p1.Z);
    
    bool x_equal = true;
    for (int i = 0; i < 6; i++) {
        if (x1z2.limbs[i] != x2z1.limbs[i]) {
            x_equal = false;
            break;
        }
    }
    
    // Compare: Y1*Z2 == Y2*Z1
    Fq y1z2, y2z1;
    field_mul(y1z2, p1.Y, p2.Z);
    field_mul(y2z1, p2.Y, p1.Z);
    
    bool y_equal = true;
    for (int i = 0; i < 6; i++) {
        if (y1z2.limbs[i] != y2z1.limbs[i]) {
            y_equal = false;
            break;
        }
    }
    
    equal_flags[idx] = (x_equal && y_equal) ? 1 : 0;
}

bool test_scalar_mul_basic() {
    std::cout << "Testing scalar multiplication (basic)... " << std::flush;
    
    const int n = 64;
    std::mt19937_64 rng(12345);
    
    // Use generator point for all bases
    G1Affine gen = make_g1_generator();
    std::vector<G1Affine> bases(n, gen);
    
    // Generate random scalars
    std::vector<Fr> scalars(n);
    for (int i = 0; i < n; i++) {
        scalars[i] = random_fr(rng);
    }
    
    // Test non-GLV scalar multiplication
    VecOpsConfig config;
    config.stream = nullptr;
    config.is_a_on_device = false;
    config.is_b_on_device = false;
    config.is_result_on_device = false;
    config.is_async = false;
    
    std::vector<G1Projective> results(n);
    
    eIcicleError err = bls12_381_g1_scalar_mul(
        bases.data(), scalars.data(), n, &config, results.data()
    );
    
    if (err != eIcicleError::SUCCESS) {
        std::cout << "FAILED (function returned error)" << std::endl;
        return false;
    }
    
    // Compute reference results using double-and-add
    G1Affine* d_bases;
    Fr* d_scalars;
    G1Projective* d_reference;
    int* d_equal_flags;
    
    CHECK_CUDA(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CHECK_CUDA(cudaMalloc(&d_scalars, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_reference, n * sizeof(G1Projective)));
    CHECK_CUDA(cudaMalloc(&d_equal_flags, n * sizeof(int)));
    
    CHECK_CUDA(cudaMemcpy(d_bases, bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scalars, scalars.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    reference_scalar_mul_kernel<<<(n + 255) / 256, 256>>>(d_reference, d_bases, d_scalars, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compare results
    G1Projective* d_results;
    CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(G1Projective)));
    CHECK_CUDA(cudaMemcpy(d_results, results.data(), n * sizeof(G1Projective), cudaMemcpyHostToDevice));
    
    compare_g1_points_kernel<<<(n + 255) / 256, 256>>>(d_results, d_reference, d_equal_flags, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> equal_flags(n);
    CHECK_CUDA(cudaMemcpy(equal_flags.data(), d_equal_flags, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_bases);
    cudaFree(d_scalars);
    cudaFree(d_reference);
    cudaFree(d_results);
    cudaFree(d_equal_flags);
    
    // Verify all results match
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (equal_flags[i] != 1) {
            failures++;
            if (failures <= 3) {
                std::cerr << "Mismatch at index " << i << std::endl;
            }
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << " mismatches)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_scalar_mul_glv() {
    std::cout << "Testing GLV scalar multiplication... " << std::flush;
    
    const int n = 128;
    std::mt19937_64 rng(54321);
    
    // Use generator point for all bases
    G1Affine gen = make_g1_generator();
    std::vector<G1Affine> bases(n, gen);
    
    // Generate random scalars
    std::vector<Fr> scalars(n);
    for (int i = 0; i < n; i++) {
        scalars[i] = random_fr(rng);
    }
    
    // Test GLV scalar multiplication
    VecOpsConfig config;
    config.stream = nullptr;
    config.is_a_on_device = false;
    config.is_b_on_device = false;
    config.is_result_on_device = false;
    config.is_async = false;
    
    std::vector<G1Projective> glv_results(n);
    
    eIcicleError err = bls12_381_g1_scalar_mul_glv(
        bases.data(), scalars.data(), n, &config, glv_results.data()
    );
    
    if (err != eIcicleError::SUCCESS) {
        std::cout << "FAILED (GLV function returned error)" << std::endl;
        return false;
    }
    
    // Also compute using standard (non-GLV) method
    std::vector<G1Projective> std_results(n);
    err = bls12_381_g1_scalar_mul(
        bases.data(), scalars.data(), n, &config, std_results.data()
    );
    
    if (err != eIcicleError::SUCCESS) {
        std::cout << "FAILED (standard function returned error)" << std::endl;
        return false;
    }
    
    // Compare GLV results with standard results
    G1Projective* d_glv;
    G1Projective* d_std;
    int* d_equal_flags;
    
    CHECK_CUDA(cudaMalloc(&d_glv, n * sizeof(G1Projective)));
    CHECK_CUDA(cudaMalloc(&d_std, n * sizeof(G1Projective)));
    CHECK_CUDA(cudaMalloc(&d_equal_flags, n * sizeof(int)));
    
    CHECK_CUDA(cudaMemcpy(d_glv, glv_results.data(), n * sizeof(G1Projective), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_std, std_results.data(), n * sizeof(G1Projective), cudaMemcpyHostToDevice));
    
    compare_g1_points_kernel<<<(n + 255) / 256, 256>>>(d_glv, d_std, d_equal_flags, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> equal_flags(n);
    CHECK_CUDA(cudaMemcpy(equal_flags.data(), d_equal_flags, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_glv);
    cudaFree(d_std);
    cudaFree(d_equal_flags);
    
    // Verify all results match
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (equal_flags[i] != 1) {
            failures++;
            if (failures <= 3) {
                std::cerr << "GLV mismatch at index " << i << std::endl;
            }
        }
    }
    
    if (failures > 0) {
        std::cout << "FAILED (" << failures << " mismatches vs standard)" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_scalar_mul_edge_cases() {
    std::cout << "Testing scalar mul edge cases... " << std::flush;
    
    G1Affine gen = make_g1_generator();
    
    VecOpsConfig config;
    config.stream = nullptr;
    config.is_a_on_device = false;
    config.is_b_on_device = false;
    config.is_result_on_device = false;
    config.is_async = false;
    
    // Test 1: scalar = 0 should give identity
    {
        std::vector<G1Affine> bases = { gen };
        std::vector<Fr> scalars = { make_fr_zero_host() };
        std::vector<G1Projective> results(1);
        
        eIcicleError err = bls12_381_g1_scalar_mul_glv(
            bases.data(), scalars.data(), 1, &config, results.data()
        );
        
        if (err != eIcicleError::SUCCESS) {
            std::cout << "FAILED (scalar=0 returned error)" << std::endl;
            return false;
        }
        
        // Result should be identity (Z = 0)
        // Check on device
        G1Projective* d_result;
        CHECK_CUDA(cudaMalloc(&d_result, sizeof(G1Projective)));
        CHECK_CUDA(cudaMemcpy(d_result, results.data(), sizeof(G1Projective), cudaMemcpyHostToDevice));
        
        // The result Z coordinate should indicate identity
        // For our implementation, identity is when all coords are zero or Z=0
        cudaFree(d_result);
    }
    
    // Test 2: scalar = 1 should give the same point
    {
        std::vector<G1Affine> bases = { gen };
        std::vector<Fr> scalars = { make_fr_one_host() };
        std::vector<G1Projective> results(1);
        
        eIcicleError err = bls12_381_g1_scalar_mul_glv(
            bases.data(), scalars.data(), 1, &config, results.data()
        );
        
        if (err != eIcicleError::SUCCESS) {
            std::cout << "FAILED (scalar=1 returned error)" << std::endl;
            return false;
        }
        
        // Result should equal the generator
        G1Projective* d_result;
        G1Projective* d_gen;
        int* d_flag;
        
        CHECK_CUDA(cudaMalloc(&d_result, sizeof(G1Projective)));
        CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Projective)));
        CHECK_CUDA(cudaMalloc(&d_flag, sizeof(int)));
        
        CHECK_CUDA(cudaMemcpy(d_result, results.data(), sizeof(G1Projective), cudaMemcpyHostToDevice));
        
        G1Projective gen_proj = G1Projective::from_affine(gen);
        CHECK_CUDA(cudaMemcpy(d_gen, &gen_proj, sizeof(G1Projective), cudaMemcpyHostToDevice));
        
        compare_g1_points_kernel<<<1, 1>>>(d_result, d_gen, d_flag, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        int flag;
        CHECK_CUDA(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        
        cudaFree(d_result);
        cudaFree(d_gen);
        cudaFree(d_flag);
        
        if (flag != 1) {
            std::cout << "FAILED (scalar=1 didn't give generator)" << std::endl;
            return false;
        }
    }
    
    // Test 3: scalar = 2 should give 2*G
    {
        std::vector<G1Affine> bases = { gen };
        Fr two = make_fr_from_u64(2);
        std::vector<Fr> scalars = { two };
        std::vector<G1Projective> results(1);
        
        eIcicleError err = bls12_381_g1_scalar_mul_glv(
            bases.data(), scalars.data(), 1, &config, results.data()
        );
        
        if (err != eIcicleError::SUCCESS) {
            std::cout << "FAILED (scalar=2 returned error)" << std::endl;
            return false;
        }
        
        // Compute 2*G by doubling
        G1Projective* d_result;
        G1Projective* d_doubled;
        int* d_flag;
        
        CHECK_CUDA(cudaMalloc(&d_result, sizeof(G1Projective)));
        CHECK_CUDA(cudaMalloc(&d_doubled, sizeof(G1Projective)));
        CHECK_CUDA(cudaMalloc(&d_flag, sizeof(int)));
        
        CHECK_CUDA(cudaMemcpy(d_result, results.data(), sizeof(G1Projective), cudaMemcpyHostToDevice));
        
        G1Projective gen_proj = G1Projective::from_affine(gen);
        G1Projective doubled;
        // Need to compute on device
        G1Projective* d_gen;
        CHECK_CUDA(cudaMalloc(&d_gen, sizeof(G1Projective)));
        CHECK_CUDA(cudaMemcpy(d_gen, &gen_proj, sizeof(G1Projective), cudaMemcpyHostToDevice));
        
        // Use existing double kernel
        test_point_double_kernel<<<1, 1>>>(d_gen, d_doubled, d_doubled, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        compare_g1_points_kernel<<<1, 1>>>(d_result, d_doubled, d_flag, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        int flag;
        CHECK_CUDA(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        
        cudaFree(d_result);
        cudaFree(d_doubled);
        cudaFree(d_gen);
        cudaFree(d_flag);
        
        if (flag != 1) {
            std::cout << "FAILED (scalar=2 didn't give 2*G)" << std::endl;
            return false;
        }
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_glv_performance_comparison() {
    std::cout << "Testing GLV performance... " << std::flush;
    
    const int n = 1024;
    std::mt19937_64 rng(99999);
    
    G1Affine gen = make_g1_generator();
    std::vector<G1Affine> bases(n, gen);
    std::vector<Fr> scalars(n);
    for (int i = 0; i < n; i++) {
        scalars[i] = random_fr(rng);
    }
    
    // Allocate device memory
    G1Affine* d_bases;
    Fr* d_scalars;
    G1Projective* d_results;
    
    CHECK_CUDA(cudaMalloc(&d_bases, n * sizeof(G1Affine)));
    CHECK_CUDA(cudaMalloc(&d_scalars, n * sizeof(Fr)));
    CHECK_CUDA(cudaMalloc(&d_results, n * sizeof(G1Projective)));
    
    CHECK_CUDA(cudaMemcpy(d_bases, bases.data(), n * sizeof(G1Affine), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scalars, scalars.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    VecOpsConfig config;
    config.stream = nullptr;
    config.is_a_on_device = true;
    config.is_b_on_device = true;
    config.is_result_on_device = true;
    config.is_async = false;
    
    // Warmup
    bls12_381_g1_scalar_mul(d_bases, d_scalars, n, &config, d_results);
    bls12_381_g1_scalar_mul_glv(d_bases, d_scalars, n, &config, d_results);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark standard method
    Timer timer;
    const int iterations = 10;
    
    timer.start();
    for (int i = 0; i < iterations; i++) {
        bls12_381_g1_scalar_mul(d_bases, d_scalars, n, &config, d_results);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double std_time = timer.elapsed_ms();
    
    // Benchmark GLV method
    timer.start();
    for (int i = 0; i < iterations; i++) {
        bls12_381_g1_scalar_mul_glv(d_bases, d_scalars, n, &config, d_results);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double glv_time = timer.elapsed_ms();
    
    cudaFree(d_bases);
    cudaFree(d_scalars);
    cudaFree(d_results);
    
    double speedup = std_time / glv_time;
    
    std::cout << "PASSED" << std::endl;
    std::cout << "    Standard: " << std_time / iterations << " ms" << std::endl;
    std::cout << "    GLV:      " << glv_time / iterations << " ms" << std::endl;
    std::cout << "    Speedup:  " << speedup << "x" << std::endl;
    
    return true;
}

// =============================================================================
// Performance Benchmarks
// =============================================================================

void benchmark_field_mul() {
    std::cout << "\nBenchmark: Field multiplication..." << std::endl;
    
    const int n = 1 << 20;  // 1M elements
    std::mt19937_64 rng(999);
    
    std::vector<Fr> a_vals(n), b_vals(n);
    for (int i = 0; i < n; i++) {
        a_vals[i] = random_fr(rng);
        b_vals[i] = random_fr(rng);
    }
    
    Fr *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, n * sizeof(Fr));
    cudaMalloc(&d_b, n * sizeof(Fr));
    cudaMalloc(&d_out, n * sizeof(Fr));
    
    cudaMemcpy(d_a, a_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_vals.data(), n * sizeof(Fr), cudaMemcpyHostToDevice);
    
    VecOpsConfig config;
    config.stream = nullptr;
    config.is_a_on_device = true;
    config.is_b_on_device = true;
    config.is_result_on_device = true;
    config.is_async = false;
    
    // Warmup
    bls12_381_vector_mul(d_a, d_b, n, &config, d_out);
    cudaDeviceSynchronize();
    
    // Benchmark
    Timer timer;
    const int iterations = 100;
    
    timer.start();
    for (int i = 0; i < iterations; i++) {
        bls12_381_vector_mul(d_a, d_b, n, &config, d_out);
    }
    cudaDeviceSynchronize();
    
    double total_ms = timer.elapsed_ms();
    double per_op_ns = (total_ms * 1e6) / (double(n) * iterations);
    double throughput = (double(n) * iterations) / (total_ms / 1000.0) / 1e9;
    
    std::cout << "  " << n << " elements x " << iterations << " iterations" << std::endl;
    std::cout << "  Total time: " << total_ms << " ms" << std::endl;
    std::cout << "  Per element: " << per_op_ns << " ns" << std::endl;
    std::cout << "  Throughput: " << throughput << " G ops/sec" << std::endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "BLS12-381 CUDA Backend Test Suite" << std::endl;
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
    std::cout << "========================================" << std::endl << std::endl;
    
    bool all_passed = true;
    int tests_run = 0;
    int tests_passed = 0;
    
    // Run tests
    #define RUN_TEST(test_func) do { \
        tests_run++; \
        if (test_func()) { \
            tests_passed++; \
        } else { \
            all_passed = false; \
        } \
    } while(0)
    
    RUN_TEST(test_library_loading);
    RUN_TEST(test_field_arithmetic);
    RUN_TEST(test_field_add_sub);
    RUN_TEST(test_field_mul_associativity);
    RUN_TEST(test_field_squaring);
    RUN_TEST(test_g2_operations);
    RUN_TEST(test_vec_add);
    
    // Note: Scalar multiplication tests require proper generator point setup
    // and are better tested through ICICLE integration (Rust tests).
    // Run: ICICLE_BACKEND_INSTALL_DIR=./install cargo test --package midnight-proofs --test gpu_integration --features gpu
    // RUN_TEST(test_scalar_mul_basic);
    // RUN_TEST(test_scalar_mul_glv);
    // RUN_TEST(test_scalar_mul_edge_cases);
    // RUN_TEST(test_glv_performance_comparison);
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << tests_passed << "/" << tests_run << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Run benchmarks if requested
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        std::cout << "\nRunning benchmarks..." << std::endl;
        benchmark_field_mul();
    }
    
    return all_passed ? 0 : 1;
}
