//
// Created by muham on 2.05.2026.
//

#ifndef CORTEXMIND_PLATFORM_SIMD_HPP
#define CORTEXMIND_PLATFORM_SIMD_HPP

#if !defined(__AVX2__)
    #if defined(_MSC_VER)
        #pragma message("No compile flag for AVX2. Compile your code with '/arch:AVX2' compile flags")
    #elif defined(__GNUC__) || defined(__clang__) //#if defined(_MSC_VER)
        #warning "No compile flag for AVX2. Compile your code with '-mavx2 -mfma' compile flags"
    #endif //#elif defined(__GNUC__) || defined(__clang__)
#endif //#if !defined(__AVX2__)

#endif //CORTEXMIND_PLATFORM_SIMD_HPP