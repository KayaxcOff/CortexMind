//
// Created by muham on 12.12.2025.
//

#ifndef CORTEXMIND_VARIABLE_HPP
#define CORTEXMIND_VARIABLE_HPP

#include <immintrin.h>
#include <cstdint>

namespace cortex::_fw::avx2 {
    #if defined(__AVX512F__)
        using mask8 = __mmask8;
    #elif defined(__AVX2__)
        using mask8 = uint8_t;
    #else
        #error "AVX2 or AVX-512 support is required"
    #endif

    using reg = __m256;
    using regi = __m256i;
    using regd = __m512d;
}

#endif //CORTEXMIND_VARIABLE_HPP