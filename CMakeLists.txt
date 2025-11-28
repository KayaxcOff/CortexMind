cmake_minimum_required(VERSION 4.0)
project(CortexMind CXX)

set(CMAKE_CXX_STANDARD 20)

add_library(CortexMind STATIC
        src/CortexMind/Utils/MathTools/Objects/vector.cpp
        src/CortexMind/Utils/MathTools/Objects/tensor.cpp
        src/CortexMind/Mind/Loss/mse.cpp
        src/CortexMind/Mind/Loss/mae.cpp
        src/CortexMind/Mind/Layers/dense.cpp
        src/CortexMind/Mind/Layers/conv.cpp
        src/CortexMind/Mind/Activation/relu.cpp
        src/CortexMind/Mind/Optimizers/sgd.cpp
        src/CortexMind/DataPrep/tokenizer.cpp
        src/CortexMind/DataPrep/image_data.cpp
        src/CortexMind/DataPrep/data.cpp
        src/CortexMind/Mind/Layers/batch_norm.cpp
        src/CortexMind/Mind/Activation/tanh.cpp
)

target_include_directories(CortexMind PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

add_executable(MindTest
        test/main.cpp
)

add_test(NAME CortexMindTest COMMAND MindTest)