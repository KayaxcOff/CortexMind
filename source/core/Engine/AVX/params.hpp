//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP

#include <immintrin.h>

namespace cortex::_fw::avx2 {

    /// @brief 256-bit vector containing 8 single-precision floats
    using vec8f = __m256;

    /// @brief 256-bit vector containing 8 32-bit integers
    using vec8i = __m256i;

    /// @brief 256-bit vector containing 4 double-precision floats
    using vec4d = __m256d;

    /// @brief 128-bit vector containing 4 single-precision floats
    using vec4f = __m128;

    /// @brief 128-bit vector containing 4 32-bit integers
    using vec4i = __m128i;

} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP