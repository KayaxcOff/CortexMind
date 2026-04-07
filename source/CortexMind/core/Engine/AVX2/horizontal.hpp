//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_HORIZONTAL_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_HORIZONTAL_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Provides horizontal operations on a single __m256 vector.
     *
     * This struct contains functions that reduce a 256-bit vector (__m256)
     * to a scalar value by applying operations across all elements.
     * Horizontal operations include sum, max, min, dot product, and mean.
     */
    struct horizontal {
        /**
         * @brief Computes the horizontal sum of all elements in a __m256 vector.
         * @param x Input vector.
         * @return Scalar value equal to the sum of all elements.
         * @note Implemented by splitting into two 128-bit halves and reducing.
         */
        [[nodiscard]]
        static __forceinline f32 sum(const vec8f x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }
        /**
         * @brief Computes the horizontal maximum of all elements in a __m256 vector.
         * @param x Input vector.
         * @return Scalar value equal to the maximum element.
         * @note Uses pairwise reduction after splitting into 128-bit halves.
         */
        [[nodiscard]]
        static __forceinline f32 max(const vec8f x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f max = _mm_max_ps(lo, hi);
            max = _mm_max_ps(max, _mm_movehl_ps(max, max));
            max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
            return _mm_cvtss_f32(max);
        }
        /**
         * @brief Computes the horizontal minimum of all elements in a __m256 vector.
         * @param x Input vector.
         * @return Scalar value equal to the minimum element.
         * @note Uses pairwise reduction after splitting into 128-bit halves.
         */
        [[nodiscard]]
        static __forceinline f32 min(const vec8f x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f min_val = _mm_min_ps(lo, hi);
            min_val = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
            min_val = _mm_min_ss(min_val, _mm_shuffle_ps(min_val, min_val, 1));
            return _mm_cvtss_f32(min_val);
        }
        /**
         * @brief Computes the dot product of two __m256 vectors.
         * @param Xx First input vector.
         * @param Xy Second input vector.
         * @return Scalar value equal to sum(Xx[i] * Xy[i]) for i = 0..7.
         * @note Implemented using element-wise multiplication followed by horizontal sum.
         */
        [[nodiscard]]
        static __forceinline f32 dot(const vec8f Xx, const vec8f Xy) {
            return sum(_mm256_mul_ps(Xx, Xy));
        }
        /**
         * @brief Computes the horizontal mean (average) of all elements in a __m256 vector.
         * @param x Input vector.
         * @return Scalar value equal to the mean of all elements.
         * @note Calculated as sum(x) / 8.0f since vec8f has 8 elements.
         */
        [[nodiscard]]
        static __forceinline f32 mean(const vec8f x) {
            return sum(x) * (1.0f / 8.0f);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_HORIZONTAL_HPP