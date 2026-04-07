//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_RUNTIME_COMPILE_FLAGS_CHECK_HPP
#define CORTEXMIND_RUNTIME_COMPILE_FLAGS_CHECK_HPP

#if !defined(__AVX2__)
    #if defined(_MSC_VER)
        #pragma message("No compile flag for AVX2. Compile your code with '/arch:AVX2' compile flags")
    #elif defined(__GNUC__) || defined(__clang__) //#if defined(_MSC_VER)
        #warning "No compile flag for AVX2. Compile your code with '-mavx2 -mfma' compile flags"
    #endif //#elif defined(__GNUC__) || defined(__clang__)
#endif //#if !defined(__AVX2__)

#if !defined(CXM_IS_CUDA_AVAILABLE)
    #if defined(_MSC_VER)
        #pragma message("No compile flag for CUDA. Compile your code with '-arch=sm_86 --extended-lambda' and use CUDA 13.2")
    #elif (defined(__GNUC__) || defined(__clang__)) && defined(_WIN32) //#if defined(_MSC_VER)
        #warning "You must use Microsoft Visual Studio Compiler to using CUDA on Windows"
    #endif //#elif (defined(__GNUC__) || defined(__clang__) ) && defined(_WIN32)
#endif //#if !defined(CXM_IS_CUDA_AVAILABLE)

#endif //CORTEXMIND_RUNTIME_COMPILE_FLAGS_CHECK_HPP