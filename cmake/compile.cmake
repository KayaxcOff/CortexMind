set(CXM_ALL_TARGETS
        CortexMind
        CXM_TEST
        CXM_G_TEST
        CXM_CUDA_G_TEST
)

if(CXM_CUDA_AVAILABLE)
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )
endif()

foreach(target IN LISTS CXM_ALL_TARGETS)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
    elseif(MSVC)
        target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /EHsc>
        )
    endif()

endforeach()