cmake_minimum_required(VERSION 4.1)
project(CortexMind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(CortexMind STATIC
        source/CortexMind/core/Engine/AVX/partial.cpp
        source/CortexMind/core/Engine/AVX/matrix.cpp
        source/CortexMind/core/Engine/Memory/alloc.cpp
        source/CortexMind/core/Engine/Storage/stor.cpp
        source/CortexMind/core/Engine/Tensor/tensor.cpp
        source/CortexMind/core/Engine/Tensor/operators.cpp
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CortexMind PRIVATE -mavx2 -mfma)
elseif(MSVC)
    target_compile_options(CortexMind PRIVATE /arch:AVX2)
endif()

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