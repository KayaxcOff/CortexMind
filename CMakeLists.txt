include(FetchContent)

cmake_minimum_required(VERSION 4.1)
project(CortexMind LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/heads/develop.zip
)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)

FetchContent_MakeAvailable(googletest nlohmann_json stb)

option(CXM_USE_CUDA "Enable CUDA backend" ON)

if (CXM_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
endif ()

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: enabling CUDA backend")
    set(CXM_CUDA_AVAILABLE TRUE)
else()
    message(STATUS "CUDA not found: disabling CUDA backend")
    set(CXM_CUDA_AVAILABLE FALSE)
endif()

if(CXM_CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    message(STATUS "CortexMind: CUDA enabled (SM ${CMAKE_CUDA_ARCHITECTURES})")
endif()

add_library(CortexMind STATIC
        source/CortexMind/core/Engine/AVX2/partial.cpp
        source/CortexMind/core/Engine/AVX2/scalar.cpp
        source/CortexMind/core/Engine/AVX2/reduce.cpp
        source/CortexMind/core/Engine/AVX2/matrix.cpp
        source/CortexMind/core/Engine/AVX2/activation.cpp
        source/CortexMind/framework/Tools/err.cpp
        source/CortexMind/runtime/ctx.cpp
        source/CortexMind/framework/Benchmark/pref.cpp
        source/CortexMind/framework/Tools/benchmark_utils.cpp
        source/CortexMind/framework/Tools/device_as_string.cpp
        source/CortexMind/tools/version.cpp
        source/CortexMind/tools/cpp_version.cpp
        source/CortexMind/tools/is_cuda_available.cpp
)

if(CXM_CUDA_AVAILABLE)
    target_sources(CortexMind PRIVATE
            source/CortexMind/core/Engine/CUDA/scalar.cu
            source/CortexMind/core/Engine/CUDA/matrix.cu
            source/CortexMind/core/Engine/CUDA/reduce.cu
            source/CortexMind/core/Engine/CUDA/activation.cu
            source/CortexMind/core/Engine/CUDA/softmax.cu
    )
endif()

target_link_libraries(CortexMind PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(CortexMind PUBLIC
        ${stb_SOURCE_DIR}
)

if(CXM_CUDA_AVAILABLE)
    target_link_libraries(CortexMind
            PRIVATE CUDA::cudart
            PRIVATE CUDA::cublas
            PRIVATE CUDA::curand
    )
endif()

target_compile_definitions(CortexMind
        PUBLIC
        $<$<BOOL:${CXM_CUDA_AVAILABLE}>:CXM_IS_CUDA_AVAILABLE=1>
        $<$<BOOL:${CXM_CUDA_AVAILABLE}>:CXM_CUDA_ARCH=86>
)

if(CXM_CUDA_AVAILABLE AND CMAKE_CUDA_COMPILER)
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 --extended-lambda>
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
    )
elseif(MSVC)
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>: /arch:AVX2 /EHsc>
    )
endif()

enable_testing()

add_executable(CXM_TEST test/main_test.cpp)

add_executable(CXM_G_TEST test/g_test.cpp)

target_include_directories(CortexMind
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/source
)

target_link_libraries(CXM_TEST PRIVATE
        CortexMind
)
target_link_libraries(CXM_G_TEST PRIVATE
        CortexMind
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(CXM_G_TEST)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CXM_TEST PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
    )
elseif(MSVC)
    target_compile_options(CXM_TEST PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /EHsc>
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CXM_G_TEST PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
    )
elseif(MSVC)
    target_compile_options(CXM_G_TEST PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /EHsc>
    )
endif()