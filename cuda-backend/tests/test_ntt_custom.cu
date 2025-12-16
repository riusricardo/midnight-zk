#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "field.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"

using namespace bls12_381;
using namespace ntt;
using namespace icicle;

// Root of unity for BLS12-381 scalar field (Fr)
// From curves/src/fq.rs (which is actually Fr)
// 0xb9b5_8d8c_5f0e_466a, 0x5b1b_4c80_1819_d7ec, 0x0af5_3ae3_52a3_1e64, 0x5bf3_adda_19e9_b27b
__device__ __constant__ uint64_t ROOT_OF_UNITY_LIMBS[4] = {
    0xb9b58d8c5f0e466aULL,
    0x5b1b4c801819d7ecULL,
    0x0af53ae352a31e64ULL,
    0x5bf3adda19e9b27bULL
};

Fr get_root_of_unity() {
    Fr root;
    root.limbs[0] = 0xb9b58d8c5f0e466aULL;
    root.limbs[1] = 0x5b1b4c801819d7ecULL;
    root.limbs[2] = 0x0af53ae352a31e64ULL;
    root.limbs[3] = 0x5bf3adda19e9b27bULL;
    return root;
}

// Helper to generate random field elements
Fr random_fr(std::mt19937_64& rng) {
    Fr r;
    for (int i = 0; i < 4; i++) {
        r.limbs[i] = rng();
    }
    // Note: This doesn't ensure it's < modulus, but for NTT linearity check it might be fine
    // strictly speaking we should reduce, but let's assume it works for now or use a proper generator
    // For correctness, we should mask or reduce.
    // Fr modulus is ~255 bits.
    r.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL; // Ensure < 2^255
    return r;
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_ICICLE(call) do { \
    eIcicleError err = call; \
    if (err != eIcicleError::SUCCESS) { \
        std::cerr << "ICICLE error at " << __FILE__ << ":" << __LINE__ << " Code: " << (int)err << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "Running NTT Custom Test..." << std::endl;

    // Initialize Domain
    Fr root = get_root_of_unity();
    NTTInitDomainConfig init_config = icicle::default_ntt_init_domain_config();
    init_config.max_log_size = 14; // Increase to test larger sizes
    CHECK_ICICLE(init_domain_cuda(root, init_config));
    
    // Test multiple sizes to verify both code paths:
    // - size <= 1024: shared memory kernel
    // - size > 1024: fused 2-stage butterfly kernel
    int test_sizes[] = {4, 8, 10, 11, 12, 13};  // 2^4=16, 2^10=1024, 2^11=2048, 2^12=4096, 2^13=8192
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int t = 0; t < num_tests; t++) {
        const int log_size = test_sizes[t];
        const int n = 1 << log_size;
        
        std::cout << "Testing size 2^" << log_size << " (" << n << " elements)";
        if (n <= 1024) {
            std::cout << " [shared memory kernel]";
        } else {
            std::cout << " [fused butterfly kernel]";
        }
        std::cout << "... " << std::flush;
    
        // Use simple known values that are already in Montgomery form
        // Fr::one() returns the Montgomery representation of 1
        std::vector<Fr> input(n);
        Fr one = Fr::one_host();
        for (int i = 0; i < n; i++) {
            input[i] = one;
        }
        
        Fr *d_input, *d_output;
        CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(Fr)));
        CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(Fr)));
        
        CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));
        
        // Forward NTT
        NTTConfig config = default_ntt_config();
        config.are_inputs_on_device = true;
        config.are_outputs_on_device = true;
        
        CHECK_ICICLE(ntt_cuda(d_input, n, NTTDir::kForward, config, d_output));
        
        // Inverse NTT
        CHECK_ICICLE(ntt_cuda(d_output, n, NTTDir::kInverse, config, d_input));
        
        // Verify result
        std::vector<Fr> result(n);
        CHECK_CUDA(cudaMemcpy(result.data(), d_input, n * sizeof(Fr), cudaMemcpyDeviceToHost));
        
        int failures = 0;
        for (int i = 0; i < n; i++) {
            bool equal = true;
            for (int j = 0; j < 4; j++) {
                if (input[i].limbs[j] != result[i].limbs[j]) {
                    equal = false;
                    break;
                }
            }
            if (!equal) {
                failures++;
                if (failures < 3) {
                    std::cout << "\n  Mismatch at index " << i;
                }
            }
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        if (failures == 0) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (" << failures << " mismatches)" << std::endl;
            CHECK_ICICLE(release_domain_cuda<Fr>());
            return 1;
        }
    }  // End of test loop
    
    std::cout << "\nAll NTT sizes verified successfully!" << std::endl;
    
    CHECK_ICICLE(release_domain_cuda<Fr>());
    
    return 0;
}
