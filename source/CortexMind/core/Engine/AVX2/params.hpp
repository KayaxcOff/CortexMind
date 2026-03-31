//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_PARAMS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_PARAMS_HPP

#include <immintrin.h>

namespace cortex::_fw::avx2 {
    using vec2d = __m128d;
    using vec4f = __m128;
    using vec4i = __m128i;
    using vec4d = __m256d;
    using vec8f = __m256;
    using vec8i = __m256i;
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_PARAMS_HPP