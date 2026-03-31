//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_RUNTIME_WARNINGS_BACKEND_CHECK_HPP
#define CORTEXMIND_RUNTIME_WARNINGS_BACKEND_CHECK_HPP

#if !defined(__AVX2__)
    #if defined(_MSC_VER)
        #pragma message("No support for AVX2 instructions. Use /arch:AVX2")
    #else //#if defined(_MSC_VER)
        #warning "No support for AVX2 instructions. Use -mavx2 -mfma"
    #endif //#if defined(_MSC_VER)
#endif //#if !defined(__AVX2__)

#if !CXM_IS_CUDA_AVAILABLE
    #if defined(_MSC_VER)
        #pragma message("No support for CUDA instructions. If you have a NVIDIA GPU, you must download CUDA 13.2 for CUDA Backend")
    #else //#if defined(_MSC_VER)
        #warning "You must download CUDA 13.2 for CUDA Backend. If you have a NVIDIA GPU, you must download CUDA 13.2 for CUDA Backend"
    #endif //#if defined(_MSC_VER)

    #if CXM_CUDA_ARCH != 86
        #pragma message("Info: CUDA architecture is set to a value different than 8.6. Make sure your GPU supports it.")
    #else //#if CXM_CUDA_ARCH != 86
        #warning "Info: CUDA architecture is set to a value different than 8.6. Make sure your GPU supports it."
    #endif //#if CXM_CUDA_ARCH != 86
#endif //#if !CXM_IS_CUDA_AVAILABLE

#endif //CORTEXMIND_RUNTIME_WARNINGS_BACKEND_CHECK_HPP