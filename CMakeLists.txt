cmake_minimum_required(VERSION 4.1)
project(CortexMind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(CortexMind STATIC
        source/core/Engine/Memory/alloc.cpp
        source/net/NeuralNetwork/Dense/dense.cpp
        source/core/Engine/Tensor/tensor.cpp
        source/net/NeuralNetwork/Flatten/flatten.cpp
        source/datasets/Scale/minmax.cpp
)

target_include_directories(CortexMind
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/source
)

add_executable(CXM_TEST test/test.cpp)

target_link_libraries(CXM_TEST
        PRIVATE CortexMind
)

enable_testing()
add_test(NAME CXM_TEST COMMAND CXM_TEST)
