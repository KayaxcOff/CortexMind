cmake_minimum_required(VERSION 4.1)
project(CortexMind LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_library(stb INTERFACE)

target_include_directories(stb INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/stb/include
)

add_library(nlohmann_json INTERFACE)

target_include_directories(nlohmann_json INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/json-3.12.0/include
)

add_library(CortexMind STATIC
        # |-------------|
        # | CPU Backend |
        # |-------------|
        source/CortexMind/core/Engine/AVX2/partial.cpp
        source/CortexMind/core/Engine/AVX2/matrix.cpp
        source/CortexMind/core/Engine/AVX2/scalar.cpp
        source/CortexMind/core/Engine/AVX2/inplace.cpp
        source/CortexMind/core/Engine/AVX2/cmp.cpp
        # |-------------|
        # | GPU Backend |
        # |-------------|
        source/CortexMind/core/Engine/CUDA/scalar.cu
        source/CortexMind/core/Engine/CUDA/inplace.cu
        source/CortexMind/core/Engine/CUDA/matrix.cu
        source/CortexMind/core/Engine/CUDA/activation.cu
        source/CortexMind/core/Engine/CUDA/reduce.cu
        source/CortexMind/core/Engine/CUDA/rand.cu
        source/CortexMind/core/Engine/CUDA/broadcast.cu
        # |----------------|
        # | Core Utilities |
        # |----------------|
        source/CortexMind/core/Tools/err.cpp
        # |--------|
        # | Memory |
        # |--------|
        source/CortexMind/core/Engine/Memory/mem.cpp
        source/CortexMind/core/Engine/Memory/buf.cu
        # |--------------------|
        # | Storage for Tensor |
        # |--------------------|
        source/CortexMind/core/Engine/Storage/stor.cpp
        # |----------|
        # | Gradient |
        # |----------|
        source/CortexMind/core/Graph/flow.cpp
        source/CortexMind/core/Graph/ops.cpp
        # |--------|
        # | Tensor |
        # |--------|
        source/CortexMind/core/Engine/Tensor/tensor.cpp
        source/CortexMind/core/Engine/Tensor/operators.cpp
        # |------------|
        # | Core - Net |
        # |------------|
        source/CortexMind/core/Net/layer.cpp
        source/CortexMind/core/Net/loss.cpp
        source/CortexMind/core/Net/optimization.cpp
        # |----------|
        # | Net - NN |
        # |----------|
        source/CortexMind/net/NeuralNetwork/dense.cpp
        source/CortexMind/net/NeuralNetwork/flatten.cpp
        source/CortexMind/net/NeuralNetwork/relu.cpp
        source/CortexMind/net/NeuralNetwork/input.cpp
        # |-----------|
        # | Net - Opt |
        # |-----------|
        source/CortexMind/net/OptimizationFunctions/sgd.cpp
        source/CortexMind/net/OptimizationFunctions/adam.cpp
        # |------------|
        # | Net - Loss |
        # |------------|
        source/CortexMind/net/LossFunctions/mse.cpp
        source/CortexMind/net/NeuralNetwork/leaky_relu.cpp
        # |-----------------|
        # | Utils - Factory |
        # |-----------------|
        source/CortexMind/utils/Cast/factory.cpp
        # |-------|
        # | Tools |
        # |-------|
        source/CortexMind/tools/funcs.cpp
)

target_link_libraries(CortexMind
        PRIVATE stb
        PRIVATE nlohmann_json
        PRIVATE CUDA::cudart
        PRIVATE CUDA::cublas
        PRIVATE CUDA::curand
)

target_include_directories(CortexMind
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/source
        ${CUDAToolkit_INCLUDE_DIRS}
)

enable_testing()

add_subdirectory(third_party/googletest-1.16.0)

add_executable(CXM_TEST test/test.cpp)

target_link_libraries(CXM_TEST
        PRIVATE
        CortexMind
        GTest::gtest_main
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
    )

    target_compile_options(CXM_TEST PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
    )
elseif(MSVC)
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /EHsc>
    )

    target_compile_options(CXM_TEST PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /EHsc>
    )
endif()

include(GoogleTest)
gtest_discover_tests(CXM_TEST
        DISCOVERY_TIMEOUT 60
)