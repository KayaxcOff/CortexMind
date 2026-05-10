//
// Created by muham on 7.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Horizontal (reduction) operations for 8-wide float vectors (AVX2).
     *
     * This struct contains static inline functions that reduce a __m256 vector
     * into a single scalar value using horizontal operations.
     */
    struct horizontal {
        /**
         * @brief Computes the horizontal sum of all 8 elements in the vector.
         *
         * @param x Input 8-wide float vector.
         * @return Sum of all elements as a scalar `f32`.
         *
         * @note Uses standard AVX2 → SSE cross-lane reduction pattern.
         * @see dot(), mean()
         */
        [[nodiscard]]
        static __forceinline f32 sum(const vec8f x) noexcept {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }
        /**
         * @brief Computes the horizontal maximum of all 8 elements.
         *
         * @param x Input 8-wide float vector.
         * @return The maximum value among all 8 lanes.
         *
         * @note Uses pairwise max reduction across lanes.
         */
        [[nodiscard]]
        static __forceinline f32 max(const vec8f x) noexcept {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f max = _mm_max_ps(lo, hi);
            max = _mm_max_ps(max, _mm_movehl_ps(max, max));
            max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
            return _mm_cvtss_f32(max);
        }
        /**
         * @brief Computes the horizontal minimum of all 8 elements.
         *
         * @param x Input 8-wide float vector.
         * @return The minimum value among all 8 lanes.
         *
         * @note Uses pairwise min reduction across lanes.
         */
        [[nodiscard]]
        static __forceinline f32 min(const vec8f x) noexcept {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f min_val = _mm_min_ps(lo, hi);
            min_val = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
            min_val = _mm_min_ss(min_val, _mm_shuffle_ps(min_val, min_val, 1));
            return _mm_cvtss_f32(min_val);
        }
        /**
         * @brief Computes the dot product of two __m256 vectors.
         *
         * @param Xx First input vector.
         * @param Xy Second input vector.
         * @return Dot product result: sum(Xx[i] * Xy[i]) for i = 0..7
         *
         * @note Equivalent to `sum(_mm256_mul_ps(Xx, Xy))`
         * @see sum()
         */
        [[nodiscard]]
        static __forceinline f32 dot(const vec8f Xx, const vec8f Xy) noexcept {
            return sum(_mm256_mul_ps(Xx, Xy));
        }
        /**
         * @brief Computes the arithmetic mean (average) of all 8 elements.
         *
         * @param x Input 8-wide float vector.
         * @return Average value = sum(x) / 8.0f
         *
         * @note Fast implementation using precomputed reciprocal.
         * @see sum()
         */
        [[nodiscard]]
        static __forceinline f32 mean(const vec8f x) noexcept {
            return sum(x) * (1.0f / 8.0f);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP