/**
 * @file icicle_field_api.cu
 * @brief ICICLE field backend registration for NTT and vector operations
 * 
 * This file registers our CUDA implementations with ICICLE's dispatcher system.
 * The pattern follows ICICLE's backend registration API:
 * 
 * 1. Declare registration functions as weak symbols (provided by ICICLE core)
 * 2. Create wrapper implementations matching ICICLE's signatures
 * 3. Use REGISTER_*_BACKEND macros for static initialization
 * 
 * The actual kernel implementations are in:
 * - field_backend.cu: NTT operations (ntt_cuda, init_domain_cuda, etc.)
 * - vec_ops.cu: Vector operations (vec_add, vec_sub, vec_mul, etc.)
 * 
 * Build: Compiled together with field_backend.cu, vec_ops.cu, params.cu
 */

#include "field.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"
#include <functional>
#include <string>

using namespace bls12_381;

// Bring in commonly used icicle types explicitly to avoid ambiguity
using icicle::eIcicleError;
using icicle::Device;
using icicle::NTTDir;
using icicle::NTTInitDomainConfig;
using icicle::VecOpsConfig;

// =============================================================================
// Forward declarations to implementation functions in other .cu files
// =============================================================================

// From field_backend.cu
namespace ntt {
    template<typename Fr>
    eIcicleError ntt_cuda(
        const Fr* input,
        int size,
        NTTDir dir,
        const icicle::NTTConfig<Fr>& config,
        Fr* output);
    
    template<typename Fr>
    eIcicleError init_domain_cuda(
        const Fr& root_of_unity,
        const NTTInitDomainConfig& config);
    
    template<typename Fr>
    eIcicleError release_domain_cuda();
}

// =============================================================================
// ICICLE Registration Function Declarations (Weak Symbols)
// =============================================================================
// These are provided by ICICLE core library (libicicle_field_bls12_381.so)
// Declared as weak so we can compile standalone without ICICLE

namespace icicle {

// NTT Registration Types (matching ICICLE's backend/ntt_backend.h)
using NttImpl = std::function<eIcicleError(
    const Device& device,
    const Fr* input,
    int size,
    NTTDir dir,
    const icicle::NTTConfig<Fr>& config,
    Fr* output)>;

using NttInitDomainImpl = std::function<eIcicleError(
    const Device& device,
    const Fr& primitive_root,
    const NTTInitDomainConfig& config)>;

using NttReleaseDomainImpl = std::function<eIcicleError(
    const Device& device,
    const Fr& phantom)>;

using NttGetRouFromDomainImpl = std::function<eIcicleError(
    const Device& device,
    uint64_t logn,
    Fr* rou)>;

// VecOps Registration Types (matching ICICLE's backend/vec_ops_backend.h)
using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const Fr* scalar_a,
    const Fr* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)>;

using VectorReduceOpImpl = std::function<eIcicleError(
    const Device& device,
    const Fr* vec_a,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)>;

// NTT registration functions (weak symbols - provided by ICICLE core)
__attribute__((weak)) void register_ntt(const std::string& deviceType, NttImpl impl);
__attribute__((weak)) void register_ntt_init_domain(const std::string& deviceType, NttInitDomainImpl impl);
__attribute__((weak)) void register_ntt_release_domain(const std::string& deviceType, NttReleaseDomainImpl impl);
__attribute__((weak)) void register_ntt_get_rou_from_domain(const std::string& deviceType, NttGetRouFromDomainImpl impl);

// VecOps registration functions (weak symbols - provided by ICICLE core)
__attribute__((weak)) void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_vector_sub(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_vector_mul(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_scalar_mul_vec(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_scalar_add_vec(const std::string& deviceType, scalarVectorOpImpl impl);
__attribute__((weak)) void register_vector_sum(const std::string& deviceType, VectorReduceOpImpl impl);

} // namespace icicle

// =============================================================================
// Registration Macros
// =============================================================================

#define ICICLE_CONCAT_INNER(a, b) a##b
#define ICICLE_CONCAT(a, b) ICICLE_CONCAT_INNER(a, b)
#define ICICLE_UNIQUE(prefix) ICICLE_CONCAT(prefix, __COUNTER__)

// NTT Registration Macros
#define REGISTER_NTT_BACKEND(DEVICE_TYPE, FUNC)                        \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_ntt_) = []() -> bool {          \
            if (icicle::register_ntt) {                                \
                icicle::register_ntt(DEVICE_TYPE, FUNC);               \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

#define REGISTER_NTT_INIT_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)            \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_ntt_init_) = []() -> bool {     \
            if (icicle::register_ntt_init_domain) {                    \
                icicle::register_ntt_init_domain(DEVICE_TYPE, FUNC);   \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

#define REGISTER_NTT_RELEASE_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)         \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_ntt_rel_) = []() -> bool {      \
            if (icicle::register_ntt_release_domain) {                 \
                icicle::register_ntt_release_domain(DEVICE_TYPE, FUNC);\
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

// VecOps Registration Macros
#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                 \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_vec_add_) = []() -> bool {      \
            if (icicle::register_vector_add) {                         \
                icicle::register_vector_add(DEVICE_TYPE, FUNC);        \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)                 \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_vec_sub_) = []() -> bool {      \
            if (icicle::register_vector_sub) {                         \
                icicle::register_vector_sub(DEVICE_TYPE, FUNC);        \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)                 \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_vec_mul_) = []() -> bool {      \
            if (icicle::register_vector_mul) {                         \
                icicle::register_vector_mul(DEVICE_TYPE, FUNC);        \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

#define REGISTER_SCALAR_MUL_VEC_BACKEND(DEVICE_TYPE, FUNC)             \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_smul_vec_) = []() -> bool {     \
            if (icicle::register_scalar_mul_vec) {                     \
                icicle::register_scalar_mul_vec(DEVICE_TYPE, FUNC);    \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

#define REGISTER_SCALAR_ADD_VEC_BACKEND(DEVICE_TYPE, FUNC)             \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_sadd_vec_) = []() -> bool {     \
            if (icicle::register_scalar_add_vec) {                     \
                icicle::register_scalar_add_vec(DEVICE_TYPE, FUNC);    \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

#define REGISTER_VECTOR_SUM_BACKEND(DEVICE_TYPE, FUNC)                 \
    namespace {                                                         \
        static bool ICICLE_UNIQUE(_reg_vec_sum_) = []() -> bool {      \
            if (icicle::register_vector_sum) {                         \
                icicle::register_vector_sum(DEVICE_TYPE, FUNC);        \
            }                                                           \
            return true;                                                \
        }();                                                            \
    }

// =============================================================================
// ICICLE-Compatible Wrapper Implementations
// =============================================================================
// These wrappers add the Device& parameter that ICICLE's dispatcher expects

// Forward declarations for kernel functions from vec_ops.cu
// Must be outside anonymous namespace to match actual definitions
namespace vec_ops {
    __global__ void vec_add_kernel(Fr* out, const Fr* a, const Fr* b, int size);
    __global__ void vec_sub_kernel(Fr* out, const Fr* a, const Fr* b, int size);
    __global__ void vec_mul_kernel(Fr* out, const Fr* a, const Fr* b, int size);
    __global__ void scalar_vec_mul_kernel(Fr* out, const Fr scalar, const Fr* vec, int size);
    __global__ void scalar_vec_add_kernel(Fr* out, const Fr scalar, const Fr* vec, int size);
}

namespace {

// -----------------------------------------------------------------------------
// NTT Wrappers
// -----------------------------------------------------------------------------

eIcicleError ntt_cuda_impl(
    const Device& device,
    const Fr* input,
    int size,
    NTTDir dir,
    const icicle::NTTConfig<Fr>& config,
    Fr* output)
{
    (void)device;  // CUDA device already set, device param unused
    return ntt::ntt_cuda<Fr>(input, size, dir, config, output);
}

eIcicleError ntt_init_domain_cuda_impl(
    const Device& device,
    const Fr& primitive_root,
    const NTTInitDomainConfig& config)
{
    (void)device;
    return ntt::init_domain_cuda<Fr>(primitive_root, config);
}

eIcicleError ntt_release_domain_cuda_impl(
    const Device& device,
    const Fr& phantom)
{
    (void)device;
    (void)phantom;
    return ntt::release_domain_cuda<Fr>();
}

// -----------------------------------------------------------------------------
// Vector Operations Wrappers
// -----------------------------------------------------------------------------

// Helper function to handle device memory allocation based on config
template<typename KernelFunc>
eIcicleError run_vec_op(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output,
    KernelFunc kernel)
{
    (void)device;  // CUDA device already set
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    const Fr* d_a = a;
    const Fr* d_b = b;
    Fr* d_output = output;
    
    bool alloc_a = !config.is_a_on_device;
    bool alloc_b = b != nullptr && !config.is_b_on_device;
    bool alloc_out = !config.is_result_on_device;
    
    // Allocate device memory if input is on host
    if (alloc_a) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, a, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_a = temp;
    }
    if (alloc_b) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, b, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_b = temp;
    }
    if (alloc_out) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    kernel<<<blocks, threads, 0, stream>>>(d_output, d_a, d_b, (int)size);
    
    // Copy output back if needed
    if (alloc_out) {
        cudaMemcpyAsync(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost, stream);
    }
    
    // Sync if not async
    if (!config.is_async) {
        cudaStreamSynchronize(stream);
    }
    
    // Free temporary allocations
    if (alloc_out) cudaFree(d_output);
    if (alloc_a) cudaFree(const_cast<Fr*>(d_a));
    if (alloc_b) cudaFree(const_cast<Fr*>(d_b));
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError vector_add_cuda_impl(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    return run_vec_op(device, a, b, size, config, output, vec_ops::vec_add_kernel);
}

eIcicleError vector_sub_cuda_impl(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    return run_vec_op(device, a, b, size, config, output, vec_ops::vec_sub_kernel);
}

eIcicleError vector_mul_cuda_impl(
    const Device& device,
    const Fr* a,
    const Fr* b,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    return run_vec_op(device, a, b, size, config, output, vec_ops::vec_mul_kernel);
}

eIcicleError scalar_mul_vec_cuda_impl(
    const Device& device,
    const Fr* scalar,
    const Fr* vec,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    (void)device;  // CUDA device already set
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    const Fr* d_vec = vec;
    Fr* d_output = output;
    Fr h_scalar;
    
    bool alloc_vec = !config.is_b_on_device;  // scalar uses a, vec uses b
    bool alloc_out = !config.is_result_on_device;
    
    // Get scalar value (scalar is single element, usually on host)
    if (config.is_a_on_device) {
        cudaMemcpy(&h_scalar, scalar, sizeof(Fr), cudaMemcpyDeviceToHost);
    } else {
        h_scalar = *scalar;
    }
    
    if (alloc_vec) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, vec, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_vec = temp;
    }
    if (alloc_out) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vec_ops::scalar_vec_mul_kernel<<<blocks, threads, 0, stream>>>(
        d_output, h_scalar, d_vec, (int)size);
    
    if (alloc_out) {
        cudaMemcpyAsync(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost, stream);
    }
    
    if (!config.is_async) {
        cudaStreamSynchronize(stream);
    }
    
    if (alloc_out) cudaFree(d_output);
    if (alloc_vec) cudaFree(const_cast<Fr*>(d_vec));
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError scalar_add_vec_cuda_impl(
    const Device& device,
    const Fr* scalar,
    const Fr* vec,
    uint64_t size,
    const VecOpsConfig& config,
    Fr* output)
{
    (void)device;  // CUDA device already set
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    const Fr* d_vec = vec;
    Fr* d_output = output;
    Fr h_scalar;
    
    bool alloc_vec = !config.is_b_on_device;
    bool alloc_out = !config.is_result_on_device;
    
    if (config.is_a_on_device) {
        cudaMemcpy(&h_scalar, scalar, sizeof(Fr), cudaMemcpyDeviceToHost);
    } else {
        h_scalar = *scalar;
    }
    
    if (alloc_vec) {
        Fr* temp;
        cudaMalloc(&temp, size * sizeof(Fr));
        cudaMemcpyAsync(temp, vec, size * sizeof(Fr), cudaMemcpyHostToDevice, stream);
        d_vec = temp;
    }
    if (alloc_out) {
        cudaMalloc(&d_output, size * sizeof(Fr));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vec_ops::scalar_vec_add_kernel<<<blocks, threads, 0, stream>>>(
        d_output, h_scalar, d_vec, (int)size);
    
    if (alloc_out) {
        cudaMemcpyAsync(output, d_output, size * sizeof(Fr), cudaMemcpyDeviceToHost, stream);
    }
    
    if (!config.is_async) {
        cudaStreamSynchronize(stream);
    }
    
    if (alloc_out) cudaFree(d_output);
    if (alloc_vec) cudaFree(const_cast<Fr*>(d_vec));
    
    return cudaGetLastError() == cudaSuccess ? 
           eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

} // anonymous namespace

// =============================================================================
// Backend Registration (Static Initialization)
// =============================================================================
// These macros create static variables whose initializers call the registration
// functions when the library is loaded

REGISTER_NTT_BACKEND("CUDA", ntt_cuda_impl);
REGISTER_NTT_INIT_DOMAIN_BACKEND("CUDA", ntt_init_domain_cuda_impl);
REGISTER_NTT_RELEASE_DOMAIN_BACKEND("CUDA", ntt_release_domain_cuda_impl);

REGISTER_VECTOR_ADD_BACKEND("CUDA", vector_add_cuda_impl);
REGISTER_VECTOR_SUB_BACKEND("CUDA", vector_sub_cuda_impl);
REGISTER_VECTOR_MUL_BACKEND("CUDA", vector_mul_cuda_impl);
REGISTER_SCALAR_MUL_VEC_BACKEND("CUDA", scalar_mul_vec_cuda_impl);
REGISTER_SCALAR_ADD_VEC_BACKEND("CUDA", scalar_add_vec_cuda_impl);

// =============================================================================
// C API Exports (for standalone testing and dlsym discovery)
// =============================================================================

extern "C" {

// These symbols are for backward compatibility / standalone testing
// The primary integration path is through ICICLE's dispatcher above

eIcicleError bls12_381_field_ntt_cuda(
    const Fr* input,
    int size,
    NTTDir dir,
    const icicle::NTTConfig<Fr>* config,
    Fr* output)
{
    return ntt::ntt_cuda<Fr>(input, size, dir, *config, output);
}

eIcicleError bls12_381_field_ntt_init_domain_cuda(
    const Fr* root_of_unity,
    const NTTInitDomainConfig* config)
{
    return ntt::init_domain_cuda<Fr>(*root_of_unity, *config);
}

eIcicleError bls12_381_field_ntt_release_domain_cuda()
{
    return ntt::release_domain_cuda<Fr>();
}

} // extern "C"
