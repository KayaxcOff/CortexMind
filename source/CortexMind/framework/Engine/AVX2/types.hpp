//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP

#include <immintrin.h>

namespace cortex::_fw::avx2 {
    using vec8f = __m256;
    using vec8i = __m256i;
    using vec4d = __m256d;
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_TYPES_HPP