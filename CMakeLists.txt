cmake_minimum_required(VERSION 4.0)
project(CortexMind)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-mavx2)

add_library(CortexMind STATIC
        source/CortexMind/framework/Tensor/var.cpp
        source/CortexMind/framework/Kernel/kernel.cpp
        source/CortexMind/net/NeuralNetwork/Dense/dense.cpp
        source/CortexMind/net/NeuralNetwork/Conv2D/conv.cpp
        source/CortexMind/net/NeuralNetwork/Flatten/flatten.cpp
        source/CortexMind/net/NeuralNetwork/MaxPooling/max_pool.cpp
        source/CortexMind/net/LossFunc/MSE/mse.cpp
        source/CortexMind/models/GradientBoosting/grad_boost.cpp
)

target_compile_options(CortexMind PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mavx2 -mfma>
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
)

target_include_directories(CortexMind PUBLIC
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/include
)

add_executable(CXM_Test
        test/cxm.cpp)

target_link_libraries(CXM_Test PRIVATE CortexMind)

add_test(NAME CortexMindTest COMMAND CXM_Test)