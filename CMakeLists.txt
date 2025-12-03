cmake_minimum_required(VERSION 4.0)
project(CortexMind)

set(CMAKE_CXX_STANDARD 20)

add_library(CortexMind STATIC
        source/CortexMind/framework/Variable/var.cpp
        source/CortexMind/framework/Kernel/kernel.cpp
        source/CortexMind/net/NeuralNetwork/Dense/dense.cpp
        source/CortexMind/net/Model/Model/model.cpp
        source/CortexMind/net/NeuralNetwork/Flatten/flatten.cpp
        source/CortexMind/net/LossFunc/MSE/mse.cpp
        source/CortexMind/net/LossFunc/MAE/mae.cpp
        source/CortexMind/net/ActivationFunc/ReLU/relu.cpp
        source/CortexMind/net/ActivationFunc/Tanh/tanh.cpp
        source/CortexMind/net/ActivationFunc/LeakyReLU/leaky.cpp
        source/CortexMind/net/NeuralNetwork/Conv2D/conv.cpp
        source/CortexMind/net/NeuralNetwork/BatchNorm/batch_norm.cpp
        source/CortexMind/net/NeuralNetwork/Dropout/dropout.cpp
        source/CortexMind/net/LossFunc/CEL/cel.cpp
        source/CortexMind/net/OptimizerFunc/SGD/sgd.cpp
        source/CortexMind/net/OptimizerFunc/Adam/adam.cpp
        source/CortexMind/tools/Text/token.cpp
)

target_include_directories(CortexMind PUBLIC
        ${CMAKE_SOURCE_DIR}/source
)

add_executable(MindTest
        test/mind.cpp
)

add_test(NAME CortexMindTest COMMAND MindTest)

target_link_libraries(MindTest PRIVATE CortexMind)