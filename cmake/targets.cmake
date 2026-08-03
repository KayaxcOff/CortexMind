# =============================================================================
# CORTEXMIND TARGET CONFIGURATION
# =============================================================================

add_library(CortexMind STATIC
        source/CortexMind/framework/Type/as_string.cpp
        source/CortexMind/framework/Type/type.cpp
        source/CortexMind/framework/Type/operator.cpp
        source/CortexMind/framework/Type/size.cpp
        source/CortexMind/framework/Tools/errors.cpp
)

if(CXM_IS_CUDA_AVAILABLE)
    target_sources(CortexMind PRIVATE
            source/CortexMind/framework/Tools/errors.cu
    )
endif()

target_include_directories(CortexMind PUBLIC
        ${CMAKE_SOURCE_DIR}/include   # header file
        ${CMAKE_SOURCE_DIR}/source    # source files
)

target_link_libraries(CortexMind PUBLIC
        nlohmann_json::nlohmann_json
        TLX::TLX
)

target_include_directories(CortexMind PRIVATE ${stb_SOURCE_DIR})

if(CXM_IS_CUDA_AVAILABLE)
    target_include_directories(CortexMind PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(CortexMind PRIVATE
            CUDA::cudart
            CUDA::cublas
            CUDA::curand
    )
endif()

target_compile_definitions(CortexMind PUBLIC
        $<$<BOOL:${CXM_IS_CUDA_AVAILABLE}>:CXM_IS_CUDA_AVAILABLE=1>
        $<$<BOOL:${CXM_IS_CUDA_AVAILABLE}>:CXM_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES}>
)