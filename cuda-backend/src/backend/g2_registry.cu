/**
 * @file g2_registry.cu
 * @brief G2 MSM Backend Registration Implementation
 * 
 * ICICLE's core library doesn't include G2 registration functions
 * (register_g2_msm, register_g2_msm_precompute_bases) because G2 support
 * is optional and behind a G2_ENABLED flag.
 * 
 * Since we need G2 MSM support, we provide our own implementation of
 * these registration functions. This creates a simple dispatcher that
 * stores the registered backend implementation and calls it when needed.
 * 
 * The C API functions (bls12_381_g2_msm_cuda, etc.) call into this
 * dispatcher to find and invoke the registered backend.
 */

#include "icicle_backend_api.cuh"
#include <unordered_map>
#include <mutex>

namespace icicle {

// =============================================================================
// G2 MSM Dispatcher Registry
// =============================================================================

namespace {
    // Thread-safe registry for G2 MSM backends
    std::mutex g_g2_registry_mutex;
    std::unordered_map<std::string, MsmG2Impl> g_g2_msm_backends;
    std::unordered_map<std::string, MsmG2PreComputeImpl> g_g2_precompute_backends;
}

// Register a G2 MSM implementation for a device type
void register_g2_msm(const std::string& deviceType, MsmG2Impl impl) {
    std::lock_guard<std::mutex> lock(g_g2_registry_mutex);
    g_g2_msm_backends[deviceType] = impl;
}

// Register a G2 MSM precompute implementation for a device type
void register_g2_msm_precompute_bases(const std::string& deviceType, MsmG2PreComputeImpl impl) {
    std::lock_guard<std::mutex> lock(g_g2_registry_mutex);
    g_g2_precompute_backends[deviceType] = impl;
}

// Get the registered G2 MSM implementation for a device type
MsmG2Impl get_g2_msm_backend(const std::string& deviceType) {
    std::lock_guard<std::mutex> lock(g_g2_registry_mutex);
    auto it = g_g2_msm_backends.find(deviceType);
    if (it != g_g2_msm_backends.end()) {
        return it->second;
    }
    return nullptr;
}

// Get the registered G2 MSM precompute implementation for a device type
MsmG2PreComputeImpl get_g2_precompute_backend(const std::string& deviceType) {
    std::lock_guard<std::mutex> lock(g_g2_registry_mutex);
    auto it = g_g2_precompute_backends.find(deviceType);
    if (it != g_g2_precompute_backends.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace icicle
