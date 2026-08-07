//
// Created by muham on 5.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP

#include <CortexMind/framework/Engine/AVX2/cmp.hpp>
#include <bit>

namespace cortex::_fw::avx2 {
    /**
     * @brief Horizontal reduction operations for AVX2 vectors.
     *
     * Provides horizontal reduction algorithms such as sum, minimum,
     * maximum, mean, and index queries for supported AVX2 vector types.
     *
     * Unlike element-wise arithmetic, these operations reduce all SIMD
     * lanes into a single scalar result or lane index.
     */
    struct horizontal {
        /**
         * @brief Computes the horizontal sum of all vector elements.
         *
         * @param x Source SIMD vector.
         *
         * @return Sum of all elements.
         */
        [[nodiscard]]
        static __forceinline float sum(const vec8f& x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f sum = _mm_add_ps(hi, lo);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }
        /**
         * @brief Computes the horizontal sum of all vector elements.
         *
         * @param x Source SIMD vector.
         *
         * @return Sum of all elements.
         */
        [[nodiscard]]
        static __forceinline double sum(const vec4d& x) {
            const vec2d hi = _mm256_extractf128_pd(x, 1);
            const vec2d lo = _mm256_castpd256_pd128(x);
            vec2d sum = _mm_add_pd(hi, lo);
            sum = _mm_hadd_pd(sum, sum);
            sum = _mm_hadd_pd(sum, sum);
            return _mm_cvtsd_f64(sum);
        }

        /**
         * @brief Computes the maximum element of a SIMD vector.
         *
         * @param x Source SIMD vector.
         *
         * @return Largest element contained in the vector.
         */
        [[nodiscard]]
        static __forceinline float max(const vec8f& x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f max = _mm_max_ps(lo, hi);
            max = _mm_max_ps(max, _mm_movehl_ps(max, max));
            max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
            return _mm_cvtss_f32(max);
        }
        /**
         * @brief Computes the maximum element of a SIMD vector.
         *
         * @param x Source SIMD vector.
         *
         * @return Largest element contained in the vector.
         */
        [[nodiscard]]
        static __forceinline double max(const vec4d& x) {
            const vec2d hi = _mm256_extractf128_pd(x, 1);
            const vec2d lo = _mm256_castpd256_pd128(x);
            vec2d max = _mm_max_pd(hi, lo);
            //min_val = _mm_min_pd(min_val, _mm_movehl_pd(min_val, min_val));
            max = _mm_max_pd(max, _mm_shuffle_pd(max, max, 1));
            return _mm_cvtsd_f64(max);
        }

        /**
         * @brief Computes the minimum element of a SIMD vector.
         *
         * @param x Source SIMD vector.
         *
         * @return Smallest element contained in the vector.
         */
        [[nodiscard]]
        static __forceinline float min(const vec8f& x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f min = _mm_min_ps(lo, hi);
            min = _mm_min_ps(min, _mm_movehl_ps(min, min));
            min = _mm_min_ss(min, _mm_shuffle_ps(min, min, 1));
            return _mm_cvtss_f32(min);
        }
        /**
         * @brief Computes the minimum element of a SIMD vector.
         *
         * @param x Source SIMD vector.
         *
         * @return Smallest element contained in the vector.
         */
        [[nodiscard]]
        static __forceinline double min(const vec4d& x) {
            const vec2d hi = _mm256_extractf128_pd(x, 1);
            const vec2d lo = _mm256_castpd256_pd128(x);
            vec2d min = _mm_min_pd(hi, lo);

            min = _mm_min_pd(min, _mm_shuffle_pd(min, min, 1));
            return _mm_cvtsd_f64(min);
        }

        /**
         * @brief Computes the arithmetic mean of all vector elements.
         *
         * @param x Source SIMD vector.
         *
         * @return Mean value of the vector.
         */
        [[nodiscard]]
        static __forceinline float mean(const vec8f& x) {
            return sum(x) / 8.0f;
        }
        /**
         * @brief Computes the arithmetic mean of all vector elements.
         *
         * @param x Source SIMD vector.
         *
         * @return Mean value of the vector.
         */
        [[nodiscard]]
        static __forceinline double mean(const vec4d& x) {
            return sum(x) / 4.0;
        }

        /**
         * @brief Returns the index of the maximum element.
         *
         * If multiple elements share the maximum value, the index of the
         * first occurrence is returned.
         *
         * @param x Source SIMD vector.
         *
         * @return Zero-based index of the maximum element.
         */
        [[nodiscard]]
        static __forceinline std::int64_t argmax(const vec8f& x) {
            const float m = max(x);
            const vec8f eqmask = cmp::eq(x, _mm256_set1_ps(m));
            const std::int32_t bits = cmp::mask(eqmask);
            return std::countr_zero(static_cast<std::uint32_t>(bits));
        }
        /**
         * @brief Returns the index of the minimum element.
         *
         * If multiple elements share the minimum value, the index of the
         * first occurrence is returned.
         *
         * @param x Source SIMD vector.
         *
         * @return Zero-based index of the minimum element.
         */
        [[nodiscard]]
        static __forceinline std::int64_t argmin(const vec8f& x) {
            const float m = min(x);
            const vec8f eqmask = cmp::eq(x, _mm256_set1_ps(m));
            const std::int32_t bits = cmp::mask(eqmask);
            return std::countr_zero(static_cast<std::uint32_t>(bits));
        }

        /**
         * @brief Returns the index of the maximum element.
         *
         * If multiple elements share the maximum value, the index of the
         * first occurrence is returned.
         *
         * @param x Source SIMD vector.
         *
         * @return Zero-based index of the maximum element.
         */
        [[nodiscard]]
        static __forceinline std::int64_t argmax(const vec4d& x) {
            const double m = max(x);
            const vec4d eqmask = cmp::eq(x, _mm256_set1_pd(m));
            const std::int32_t bits = cmp::mask(eqmask);
            return std::countr_zero(static_cast<std::uint32_t>(bits));
        }
        /**
         * @brief Returns the index of the minimum element.
         *
         * If multiple elements share the minimum value, the index of the
         * first occurrence is returned.
         *
         * @param x Source SIMD vector.
         *
         * @return Zero-based index of the minimum element.
         */
        [[nodiscard]]
        static __forceinline std::int64_t argmin(const vec4d& x) {
            const double m = min(x);
            const vec4d eqmask = cmp::eq(x, _mm256_set1_pd(m));
            const std::int32_t bits = cmp::mask(eqmask);
            return std::countr_zero(static_cast<std::uint32_t>(bits));
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP