option(BUILD_TESTING "Compile test files" ON)

if(CXM_IS_CUDA_AVAILABLE)
    # for all-compile modes
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --extended-lambda           # Enables the use of C++ lambda functions within CUDA kernels.
                --expt-relaxed-constexpr    # Allows constexpr host functions to be called from CUDA device code.
            >)
    # for debug mode
    target_compile_options(CortexMind PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
                -g                #Generates debug symbols for the host (CPU) side.
                -G                # Generates debug symbols for the device (GPU / kernel) side and disables optimizations.
                -src-in-ptx       # Embeds source code lines into the PTX code during debugging.
            >)
    # for release mode
    target_compile_options(CortexMind PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
                -O3               # Highest level of optimization for NVCC and the underlying C++ compiler.
                --use_fast_math   # Runs math functions in a fast (but less precise) hardware-level mode.
            >)
endif ()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # for gcc all-compile modes
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                -mavx2            # Enables the AVX2 vector instruction set (SIMD parallelization).
                -mfma             # Enables processor support for Fused Multiply-Add (multiply and add in a single cycle).
            >)
    # for gcc debug mode
    target_compile_options(CortexMind PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:
                -O0                        # Completely disables optimizations, allowing the code to be traced line by line.
                -g                         # Generates standard debugging symbols.
                -Wall                      # Enables all commonly used warnings.
                -Wextra                    # Enables extra detailed/strict warnings.
                -Wpedantic                 # Checks for full compliance with ISO C++ standards and warns about non-standard usage.
                -fsanitize=address         # AddressSanitizer: Detects errors such as memory leaks and buffer overflows at runtime.
                -fsanitize=undefined       # UndefinedBehaviorSanitizer: Catches undefined behavior as defined by the C++ standards.
            >)
    # for gcc release mode
    target_compile_options(CortexMind PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:
                -O3                        # Enables the most aggressive compiler optimizations (loop unrolling, vectorization, etc.).
                -march=native              # Optimizes the code specifically for the architecture of the processor on which the compilation is performed (enables all additional instructions).
                -flto                      # Link-Time Optimization: It can also inline functions across different source files (.cpp).
            >)

    # link flags
    # for Sanitizers in debug mode
    target_link_options(CortexMind PRIVATE
            $<$<CONFIG:Debug>:
                -fsanitize=address,undefined
            >)
elseif (MSVC)
    # for msvc all-compile modes
    target_compile_options(CortexMind PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                /arch:AVX2        # Enables the AVX2 vector instruction set for Windows processors.
                /EHsc             # Enables the C++ standard exception handling model.
            >)
    # for msvc debug mode
    target_compile_options(CortexMind PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:
                /Od                # Disables optimizations and facilitates debugging.
                /W4                # The highest recommended logical warning level for MSVC.
                /fsanitize=address # MSVC AddressSanitizer: Detects errors such as memory leaks and overflows at runtime on Windows.
            >)
    # for msvc release mode
    target_compile_options(CortexMind PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:
                /O2               # Maximum speed optimization.
                /Oi               # It translates appropriate functions directly into processor instructions (intrinsics).
                /Ot               # It tells the compiler to prioritize speed over reducing code size.
                /GL               # Whole Program Optimization: MSVC's equivalent to LTO (Link-Time Optimization); it performs cross-file optimization.
                /fp:fast          # Fast Math Mode: Executes floating-point (float/double) operations at maximum speed by relaxing IEEE 754 standard precision rules.
            >)

    # link flags
    target_link_options(CortexMind PRIVATE
            $<$<CONFIG:Release>:
                /LTCG             # Link-Time Code Generation: Enables the /GL flag to be completed at the link stage.
            >)
endif ()