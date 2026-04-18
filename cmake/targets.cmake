add_library(CortexMind STATIC
        source/CortexMind/core/Engine/AVX2/partial.cpp
        source/CortexMind/core/Engine/AVX2/scalar.cpp
        source/CortexMind/core/Engine/AVX2/reduce.cpp
        source/CortexMind/core/Engine/AVX2/matrix.cpp
        source/CortexMind/core/Engine/AVX2/activation.cpp
        source/CortexMind/runtime/ctx.cpp
        source/CortexMind/framework/Benchmark/pref.cpp
        source/CortexMind/framework/Tools/err.cpp
        source/CortexMind/framework/Tools/benchmark_utils.cpp
        source/CortexMind/framework/Tools/device_as_string.cpp
        source/CortexMind/framework/Tools/memory_utils.cpp
        source/CortexMind/framework/Memory/mem.cpp
        source/CortexMind/framework/Gradient/flow.cpp
        source/CortexMind/framework/Storage/stor.cpp
        source/CortexMind/framework/Storage/operators.cpp
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
            source/CortexMind/framework/Memory/forge.cu
    )
endif()

target_include_directories(CortexMind PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/source
        ${stb_SOURCE_DIR}
)

target_link_libraries(CortexMind PUBLIC
        nlohmann_json::nlohmann_json
)

if(CXM_CUDA_AVAILABLE)
    target_link_libraries(CortexMind PRIVATE
            CUDA::cudart
            CUDA::cublas
            CUDA::curand
    )
endif()

target_compile_definitions(CortexMind PUBLIC
        $<$<BOOL:${CXM_CUDA_AVAILABLE}>:CXM_IS_CUDA_AVAILABLE=1>
        $<$<BOOL:${CXM_CUDA_AVAILABLE}>:CXM_CUDA_ARCH=86>
)