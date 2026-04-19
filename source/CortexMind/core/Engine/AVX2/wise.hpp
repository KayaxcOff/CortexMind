//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_WISE_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_WISE_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized element-wise mathematical functions.
     *
     * Provides high-performance implementations of common unary mathematical
     * operations using AVX2 vectorized instructions with scalar fallback
     * for remaining elements.
     */
    struct wise {
        /**
         * @brief Element-wise power operation: Z[i] = Xx[i] ^ exp
         * @param Xx  Input array
         * @param exp Scalar exponent
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void pow(const f32* __restrict Xx, f32 exp, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise square root: Z[i] = sqrt(Xx[i])
         * @param Xx  Input array
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void square(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise natural logarithm: Z[i] = log(Xx[i])
         * @param Xx  Input array
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise exponential: Z[i] = exp(Xx[i])
         * @param Xx  Input array
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_WISE_HPP