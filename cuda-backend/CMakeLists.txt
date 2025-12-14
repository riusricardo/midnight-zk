cmake_minimum_required(VERSION 3.18)
project(icicle_backend_cuda_bls12_381 LANGUAGES CXX CUDA)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architectures (Turing through Blackwell)
# 75 = Turing (RTX 20xx)
# 80 = Ampere (A100)
# 86 = Ampere (RTX 30xx)
# 89 = Ada Lovelace (RTX 40xx)
# 100 = Hopper (H100)
# 120 = Blackwell (RTX 50xx)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 100 120)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files for field library
set(FIELD_SOURCES
    src/params.cu
    src/field/vec_ops.cu
    src/field/ntt.cu
    src/field/field_backend.cu
)

# Source files for curve library  
set(CURVE_SOURCES
    src/curve/params.cu
    src/curve/point_ops.cu
    src/curve/msm.cu
    src/curve/ecntt.cu
    src/curve/montgomery.cu
    src/curve/curve_backend.cu
)

# Note: msm_kernels.cu requires CUB library for sorting
# Add it conditionally if CUB is available
find_package(CUB QUIET)
if(CUB_FOUND)
    list(APPEND CURVE_SOURCES src/curve/msm_kernels.cu)
    message(STATUS "CUB found, enabling optimized MSM kernels")
endif()

# Field library
add_library(icicle_backend_cuda_field_bls12_381 SHARED ${FIELD_SOURCES})
target_link_libraries(icicle_backend_cuda_field_bls12_381 CUDA::cudart)
set_target_properties(icicle_backend_cuda_field_bls12_381 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "icicle_backend_cuda_field_bls12_381"
)

# Curve library
add_library(icicle_backend_cuda_curve_bls12_381 SHARED ${CURVE_SOURCES})
target_link_libraries(icicle_backend_cuda_curve_bls12_381 
    icicle_backend_cuda_field_bls12_381
    CUDA::cudart
)
set_target_properties(icicle_backend_cuda_curve_bls12_381 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "icicle_backend_cuda_curve_bls12_381"
)

# Installation
install(TARGETS 
    icicle_backend_cuda_field_bls12_381 
    icicle_backend_cuda_curve_bls12_381
    LIBRARY DESTINATION lib/backend/bls12_381/cuda
)

# Tests
enable_testing()

add_executable(test_backend 
    tests/test_backend.cu
    src/params.cu  # Need device constants for separable compilation
)
target_link_libraries(test_backend 
    icicle_backend_cuda_curve_bls12_381
    icicle_backend_cuda_field_bls12_381
    CUDA::cudart
)
set_target_properties(test_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME backend_tests COMMAND test_backend)

# Comprehensive test suite with profiling
add_executable(test_vectors
    tests/test_vectors.cu
    src/params.cu
)
target_link_libraries(test_vectors
    icicle_backend_cuda_curve_bls12_381
    icicle_backend_cuda_field_bls12_381
    CUDA::cudart
)
set_target_properties(test_vectors PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME vector_tests COMMAND test_vectors)
