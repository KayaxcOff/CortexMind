//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_REDUCE_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_REDUCE_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized reduction operations on float arrays.
     *
     * Provides high-performance implementations of common reduction functions
     * such as sum, mean, variance, min/max, norms, and dot product using
     * vectorized AVX2 instructions with scalar fallback for remaining elements.
     */
    struct reduce {
        /**
         * @brief Returns the sum of all elements in the array.
         */
        static f32 sum(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the mean (average) of all elements in the array.
         */
        static f32 mean(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the variance of the array (population variance, divided by N).
         */
        static f32 var(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the standard deviation of the array.
         */
        static f32 std(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the minimum value in the array.
         */
        static f32 min(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the maximum value in the array.
         */
        static f32 max(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the L1 norm (sum of absolute values).
         */
        static f32 norm1(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the L2 norm (Euclidean norm).
         */
        static f32 norm2(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the dot product of two arrays (Xx · Xy).
         */
        static f32 dot(const f32* __restrict Xx, const f32* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_REDUCE_HPP