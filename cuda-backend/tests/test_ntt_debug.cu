#include <iostream>
#include <cuda_runtime.h>
#include "field.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"

using namespace bls12_381;
using namespace ntt;
using namespace icicle;

Fr get_root_of_unity() {
    Fr root;
    root.limbs[0] = 0xb9b58d8c5f0e466aULL;
    root.limbs[1] = 0x5b1b4c801819d7ecULL;
    root.limbs[2] = 0x0af53ae352a31e64ULL;
    root.limbs[3] = 0x5bf3adda19e9b27bULL;
    return root;
}

#define CHECK(call) do { auto e = call; if (e != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

void print_fr(const char* name, const Fr& f) {
    printf("%s: %016lx %016lx %016lx %016lx\n", name, f.limbs[3], f.limbs[2], f.limbs[1], f.limbs[0]);
}

int main() {
    // Init domain
    Fr root = get_root_of_unity();
    NTTInitDomainConfig init_cfg = default_ntt_init_domain_config();
    init_cfg.max_log_size = 4;
    
    auto err = init_domain_cuda(root, init_cfg);
    if (err != eIcicleError::SUCCESS) {
        std::cerr << "init_domain failed: " << (int)err << std::endl;
        return 1;
    }
    
    // Check domain
    Domain<Fr>* dom = Domain<Fr>::get_domain(2);
    if (!dom) {
        std::cerr << "Domain not found!" << std::endl;
        return 1;
    }
    std::cout << "Domain size: " << dom->size << ", log_size: " << dom->log_size << std::endl;
    
    // Read first few twiddles
    Fr* h_tw = new Fr[4];
    CHECK(cudaMemcpy(h_tw, dom->twiddles, 4 * sizeof(Fr), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; i++) {
        char name[32];
        sprintf(name, "twiddle[%d]", i);
        print_fr(name, h_tw[i]);
    }
    
    // Test with simple input
    const int n = 4;
    Fr h_input[n], h_output[n];
    Fr one = Fr::one_host();
    for (int i = 0; i < n; i++) h_input[i] = one;
    
    Fr *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n * sizeof(Fr)));
    CHECK(cudaMalloc(&d_out, n * sizeof(Fr)));
    CHECK(cudaMemcpy(d_in, h_input, n * sizeof(Fr), cudaMemcpyHostToDevice));
    
    NTTConfig cfg = default_ntt_config();
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    
    err = ntt_cuda(d_in, n, NTTDir::kForward, cfg, d_out);
    if (err != eIcicleError::SUCCESS) {
        std::cerr << "Forward NTT failed: " << (int)err << std::endl;
        return 1;
    }
    
    CHECK(cudaMemcpy(h_output, d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    std::cout << "\nForward NTT results:" << std::endl;
    for (int i = 0; i < n; i++) {
        char name[32];
        sprintf(name, "output[%d]", i);
        print_fr(name, h_output[i]);
    }
    
    // For all-ones input, forward NTT should give [n, 0, 0, 0] in standard ordering
    // (or Montgomery representation of that)
    
    // Now inverse
    err = ntt_cuda(d_out, n, NTTDir::kInverse, cfg, d_in);
    if (err != eIcicleError::SUCCESS) {
        std::cerr << "Inverse NTT failed: " << (int)err << std::endl;
        return 1;
    }
    
    CHECK(cudaMemcpy(h_output, d_in, n * sizeof(Fr), cudaMemcpyDeviceToHost));
    
    std::cout << "\nAfter inverse (should match input):" << std::endl;
    for (int i = 0; i < n; i++) {
        char name[32];
        sprintf(name, "result[%d]", i);
        print_fr(name, h_output[i]);
    }
    
    std::cout << "\nExpected (one in Montgomery):" << std::endl;
    print_fr("one", one);
    
    release_domain_cuda<Fr>();
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_tw;
    
    return 0;
}
