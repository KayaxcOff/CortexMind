cmake_minimum_required(VERSION 4.0)
project(CortexMind)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-mavx2)

add_library(CortexMind STATIC
        source/CortexMind/framework/Core/Tensor/tensor.cpp
        source/CortexMind/framework/Tools/Kernel/kernel.cpp
        source/CortexMind/framework/Tools/MathTools/math.cpp
        source/CortexMind/net/NeuralNetwork/Dense/dense.cpp
        source/CortexMind/net/NeuralNetwork/Conv2D/conv2d.cpp
        source/CortexMind/net/NeuralNetwork/MaxPooling/max_pool.cpp
        source/CortexMind/tools/Image/image.cpp
        source/CortexMind/tools/Tokenizer/token.cpp
        source/CortexMind/tools/Encode/encode.cpp
        source/CortexMind/net/NeuralNetwork/Embedding/embedding.cpp
        source/CortexMind/net/NeuralNetwork/Dropout/dropout.cpp
        source/CortexMind/net/ActivationFunc/Sigmoid/sigmoid.cpp
        source/CortexMind/net/ActivationFunc/ReLU/relu.cpp
        source/CortexMind/net/ActivationFunc/LeakyReLU/leaky.cpp
        source/CortexMind/net/OptimFunc/SGD/sgd.cpp
        source/CortexMind/net/OptimFunc/Momentum/momentum.cpp
        source/CortexMind/net/OptimFunc/Adam/adam.cpp
        source/CortexMind/net/NeuralNetwork/BatchNormazilation/batch_norm.cpp
)

target_compile_options(CortexMind PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mavx2 -mfma>
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
)

target_include_directories(CortexMind
        PUBLIC
            ${CMAKE_SOURCE_DIR}/source
            ${CMAKE_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_SOURCE_DIR}/third_party/stb/include
)

add_executable(CXM_Test
        test/test.cpp
)

target_link_libraries(CXM_Test PRIVATE CortexMind)

add_test(NAME CortexMindTest COMMAND CXM_Test)