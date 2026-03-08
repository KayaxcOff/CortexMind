cmake_minimum_required(VERSION 4.1)
project(CortexMind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(OpenCL::OpenCL UNKNOWN IMPORTED)

set_target_properties(OpenCL::OpenCL PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/third_party/OpenCL/lib/OpenCL.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/OpenCL"
)

add_library(stb INTERFACE)

target_include_directories(stb INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/stb
)

add_library(nlohmann_json INTERFACE)

target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/json-3.12.0/include
)

add_library(CortexMind STATIC
        # <---------- Core - AVX2 ---------->
        source/CortexMind/core/Engine/AVX2/partial.cpp
        source/CortexMind/core/Engine/AVX2/matrix.cpp
        # <---------- Core - CL2 ---------->
        source/CortexMind/core/Engine/CL2/context.cpp
        source/CortexMind/core/Engine/CL2/program.cpp
        source/CortexMind/core/Engine/CL2/registry.cpp
        # <---------- Core - Memory ---------->
        source/CortexMind/core/Engine/Memory/mem.cpp
        source/CortexMind/core/Engine/Memory/buffer.cpp
        # <---------- Core - Storage ---------->
        source/CortexMind/core/Engine/Storage/stor.cpp
        # <---------- Core - Tensor ---------->
        source/CortexMind/core/Engine/Tensor/tensor.cpp
        source/CortexMind/core/Engine/Tensor/operators.cpp
        # <---------- Core - Gradient ---------->
        source/CortexMind/core/Graph/flow_ops.cpp
        # <---------- Core - Tools ---------->
        source/CortexMind/core/Tools/err.cpp
        source/CortexMind/core/Tools/throw.cpp
        source/CortexMind/core/Tools/image.cpp
        # <---------- Core - Net ---------->
        source/CortexMind/core/Net/layer.cpp
        source/CortexMind/core/Net/optimization.cpp
        source/CortexMind/core/Net/callback.cpp
        # <---------- Core - Kernels ---------->
        source/CortexMind/core/Kernels/kernel.cpp
        # <---------- Net - Neural Network ---------->
        source/CortexMind/net/NeuralNetwork/dense.cpp
        source/CortexMind/net/NeuralNetwork/conv2d.cpp
        source/CortexMind/net/NeuralNetwork/flatten.cpp
        source/CortexMind/net/NeuralNetwork/relu.cpp
        source/CortexMind/net/NeuralNetwork/input.cpp
        source/CortexMind/net/NeuralNetwork/global_avg.cpp
        source/CortexMind/net/NeuralNetwork/tanh.cpp
        source/CortexMind/net/NeuralNetwork/sigmoid.cpp
        source/CortexMind/net/NeuralNetwork/dropout.cpp
        source/CortexMind/net/NeuralNetwork/batch_norm.cpp
        source/CortexMind/net/NeuralNetwork/leaky_relu.cpp
        # <---------- Net - Loss Functions ---------->
        source/CortexMind/net/LossFunctions/mse.cpp
        source/CortexMind/net/LossFunctions/mae.cpp
        source/CortexMind/net/LossFunctions/bce.cpp
        # <---------- Net - Optimization Functions ---------->
        source/CortexMind/net/OptimizationFunctions/sgd.cpp
        source/CortexMind/net/OptimizationFunctions/adam.cpp
        # <---------- Net - Model ---------->
        source/CortexMind/net/Model/model.cpp
        # <---------- Net - Callbacks ---------->
        source/CortexMind/net/Callbacks/early_stop.cpp
        # <---------- Utils - Cast ---------->
        source/CortexMind/utils/Cast/factory.cpp
        # <---------- Utils - Data ---------->
        source/CortexMind/utils/Data/image.cpp
        source/CortexMind/utils/Data/file.cpp
        # <---------- Utils - Metric ---------->
        source/CortexMind/utils/Metric/metric.cpp
        # <---------- Dataset - Frame ---------->
        source/CortexMind/dataset/Frame/df.cpp
        # <---------- Dataset - Manager ---------->
        source/CortexMind/dataset/Manager/data.cpp
        source/CortexMind/core/Net/loss.cpp
)

# Path of OpenCL Kernels
target_compile_definitions(CortexMind PRIVATE
        "CORTEX_KERNEL_DIR=\"${CMAKE_SOURCE_DIR}/source/CortexMind/core/Engine/CL2/Kernels\""
)

target_include_directories(CortexMind
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/source
)

target_link_libraries(CortexMind
        PUBLIC OpenCL::OpenCL
        PRIVATE stb
        PRIVATE nlohmann_json
)

enable_testing()

add_subdirectory(third_party/googletest-1.16.0)

add_executable(CXM_TEST test/test.cpp)

target_link_libraries(CXM_TEST
        PRIVATE
        CortexMind
        GTest::gtest_main
)

# Compile commands for SIMD
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CortexMind PRIVATE -mavx2 -mfma)
    target_compile_options(CXM_TEST PRIVATE -mavx2 -mfma)
elseif(MSVC)
    target_compile_options(CortexMind PRIVATE /arch:AVX2 /EHsc)
    target_compile_options(CXM_TEST PRIVATE /arch:AVX2 /EHsc)
endif()

include(GoogleTest)
gtest_discover_tests(CXM_TEST
        DISCOVERY_TIMEOUT 60
)