//
// Created by muham on 2.05.2026.
//

#ifndef CORTEXMIND_PLATFORM_CUDA_HPP
#define CORTEXMIND_PLATFORM_CUDA_HPP

#if !defined(CXM_IS_CUDA_AVAILABLE)
    #if defined(_MSC_VER)
        #pragma message("No compile flag for CUDA. Compile your code with '-arch=sm_86 --extended-lambda' and use CUDA 13.2")
    #elif (defined(__GNUC__) || defined(__clang__)) && defined(_WIN32) //#if defined(_MSC_VER)
        #warning "You must use Microsoft Visual Studio Compiler to using CUDA on Windows"
    #endif //#elif (defined(__GNUC__) || defined(__clang__) ) && defined(_WIN32)
#endif //#if !defined(CXM_IS_CUDA_AVAILABLE)

#endif //CORTEXMIND_PLATFORM_CUDA_HPP