//
// Created by muham on 2.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP

#include <immintrin.h>

namespace cortex::_fw::avx2 {

    /**
     * @brief 128-bit vector of two double-precision floating point values.
     *
     * This type is an alias for the __m128d intrinsic type, used in AVX/AVX2 operations.
     */
    using vec2d = __m128d;

    /**
     * @brief 128-bit vector of four single-precision floating point values.
     *
     * This type is an alias for the __m128 intrinsic type, used in AVX/AVX2 operations.
     */
    using vec4f = __m128;

    /**
     * @brief 128-bit vector of four 32-bit integer values.
     *
     * This type is an alias for the __m128i intrinsic type, used in AVX/AVX2 operations.
     */
    using vec4i = __m128i;

    /**
     * @brief 256-bit vector of four double-precision floating point values.
     *
     * This type is an alias for the __m256d intrinsic type, used in AVX/AVX2 operations.
     */
    using vec4d = __m256d;

    /**
     * @brief 256-bit vector of eight single-precision floating point values.
     *
     * This type is an alias for the __m256 intrinsic type, used in AVX/AVX2 operations.
     */
    using vec8f = __m256;

    /**
     * @brief 256-bit vector of eight 32-bit integer values.
     *
     * This type is an alias for the __m256i intrinsic type, used in AVX/AVX2 operations.
     */
    using vec8i = __m256i;

} // namespace cortex::_fw::avx2

#endif // CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP