//
// Created by muham on 5.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>

namespace cortex::_fw::avx2 {
    struct horizontal {
        [[nodiscard]]
        static __forceinline float sum(const vec8f& x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f sum = _mm_add_ps(hi, lo);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }
        [[nodiscard]]
        static __forceinline double sum(const vec4d& x) {
            const vec2d hi = _mm256_extractf128_pd(x, 1);
            const vec2d lo = _mm256_castpd256_pd128(x);
            vec2d sum = _mm_add_pd(hi, lo);
            sum = _mm_hadd_pd(sum, sum);
            sum = _mm_hadd_pd(sum, sum);
            return _mm_cvtsd_f64(sum);
        }

        [[nodiscard]]
        static __forceinline float max(const vec8f& x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f max = _mm_max_ps(lo, hi);
            max = _mm_max_ps(max, _mm_movehl_ps(max, max));
            max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
            return _mm_cvtss_f32(max);
        }
        [[nodiscard]]
        static __forceinline double max(const vec4d& x) {
            const vec2d hi = _mm256_extractf128_pd(x, 1);
            const vec2d lo = _mm256_castpd256_pd128(x);
            vec2d max = _mm_max_pd(hi, lo);
            //min_val = _mm_min_pd(min_val, _mm_movehl_pd(min_val, min_val));
            max = _mm_max_pd(max, _mm_shuffle_pd(max, max, 1));
            return _mm_cvtsd_f64(max);
        }

        [[nodiscard]]
        static __forceinline float min(const vec8f& x) {
            const vec4f hi = _mm256_extractf128_ps(x, 1);
            const vec4f lo = _mm256_castps256_ps128(x);
            vec4f min = _mm_min_ps(lo, hi);
            min = _mm_min_ps(min, _mm_movehl_ps(min, min));
            min = _mm_min_ss(min, _mm_shuffle_ps(min, min, 1));
            return _mm_cvtss_f32(min);
        }
        [[nodiscard]]
        static __forceinline double min(const vec4d& x) {
            const vec2d hi = _mm256_extractf128_pd(x, 1);
            const vec2d lo = _mm256_castpd256_pd128(x);
            vec2d min = _mm_min_pd(hi, lo);

            min = _mm_min_pd(min, _mm_shuffle_pd(min, min, 1));
            return _mm_cvtsd_f64(min);
        }

        [[nodiscard]]
        static __forceinline float mean(const vec8f& x) {
            return sum(x) / (1.0f / 8.0f);
        }
        [[nodiscard]]
        static __forceinline double mean(const vec4d& x) {
            return sum(x) / (1.0 / 4.0);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_HORIZONTAL_HPP