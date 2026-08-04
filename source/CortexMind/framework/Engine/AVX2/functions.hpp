//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>

namespace cortex::_fw::avx2 {
    [[nodiscard]]
    inline vec8f set1(const float value) {
        return vec8f(_mm256_set1_ps(value));
    }
    [[nodiscard]]
    inline vec8i set1(const std::int32_t value) {
        return vec8i(_mm256_set1_epi32(value));
    }
    [[nodiscard]]
    inline vec4d set1(const double value) {
        return vec4d(_mm256_set1_pd(value));
    }

    [[nodiscard]]
    inline vec8f square(const vec8f& x) {
        return x * x;
    }
    [[nodiscard]]
    inline vec8i square(const vec8i& x) {
        return x * x;
    }
    [[nodiscard]]
    inline vec4d square(const vec4d& x) {
        return x * x;
    }

    [[nodiscard]]
    inline vec8f sqrt(const vec8f& x) {
        return vec8f(_mm256_sqrt_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f sqrt(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_sqrt_ps(r0));
    }
    [[nodiscard]]
    inline vec4d sqrt(const vec4d& x) {
        return vec4d(_mm256_sqrt_pd(x.raw()));
    }

    [[nodiscard]]
    inline vec8f rsqrt(const vec8f& x) {
        return vec8f(_mm256_rsqrt_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f rsqrt(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_rsqrt_ps(r0));
    }
    [[nodiscard]]
    inline vec4d rsqrt(const vec4d& x) {
        const __m256d r1 = _mm256_sqrt_pd(x.raw());
        return vec4d(_mm256_div_pd(set1(1.0).raw(), r1));
    }

    [[nodiscard]]
    inline vec8f pow(const vec8f& x1, const vec8f& x2) {
        return vec8f(_mm256_pow_ps(x1.raw(), x2.raw()));
    }
    [[nodiscard]]
    inline vec8f pow(const vec8i& x1, const vec8i& x2) {
        const __m256 r0 = _mm256_cvtepi32_ps(x1.raw());
        const __m256 r1 = _mm256_cvtepi32_ps(x2.raw());
        return vec8f(_mm256_pow_ps(r0, r1));
    }

    [[nodiscard]]
    inline vec8f log(const vec8f& x) {
        return vec8f(_mm256_log_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f log(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_log_ps(r0));
    }
    [[nodiscard]]
    inline vec4d log(const vec4d& x) {
        return vec4d(_mm256_log_pd(x.raw()));
    }

    [[nodiscard]]
    inline vec8f exp(const vec8f& x) {
        return vec8f(_mm256_exp_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f exp(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_exp_ps(r0));
    }
    [[nodiscard]]
    inline vec4d exp(const vec4d& x) {
        return vec4d(_mm256_exp_pd(x.raw()));
    }

    [[nodiscard]]
    inline vec8f neg(const vec8f& x) {
        return vec8f(_mm256_xor_ps(x.raw(), set1(-1.0f).raw()));
    }
    [[nodiscard]]
    inline vec8i neg(const vec8i& x) {
        return vec8i(_mm256_xor_epi32(x.raw(), set1(-1).raw()));
    }
    [[nodiscard]]
    inline vec4d neg(const vec4d& x) {
        return vec4d(_mm256_xor_pd(x.raw(), set1(-1.0).raw()));
    }

    [[nodiscard]]
    inline vec8f erf(const vec8f& x) {
        return vec8f(_mm256_erf_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f erf(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_erf_ps(r0));
    }
    [[nodiscard]]
    inline vec4d erf(const vec4d& x) {
        return vec4d(_mm256_erf_pd(x.raw()));
    }

    [[nodiscard]]
    inline vec8f sin(const vec8f& x) {
        return vec8f(_mm256_sin_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f sin(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_sin_ps(r0));
    }
    [[nodiscard]]
    inline vec4d sin(const vec4d& x) {
        return vec4d(_mm256_sin_pd(x.raw()));
    }

    [[nodiscard]]
    inline vec8f cos(const vec8f& x) {
        return vec8f(_mm256_cos_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f cos(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_cos_ps(r0));
    }
    [[nodiscard]]
    inline vec4d cos(const vec4d& x) {
        return vec4d(_mm256_cos_pd(x.raw()));
    }

    [[nodiscard]]
    inline vec8f abs(const vec8f& x) {
        return vec8f(_mm256_andnot_ps(set1(-0.0f).raw(), x.raw()));
    }
    [[nodiscard]]
    inline vec8i abs(const vec8i& x) {
        return vec8i(_mm256_abs_epi32(x.raw()));
    }
    [[nodiscard]]
    inline vec4d abs(const vec4d& x) {
        return vec4d(_mm256_andnot_pd(set1(-0.0).raw(), x.raw()));
    }

    [[nodiscard]]
    inline vec8f rcp(const vec8f& x) {
        return vec8f(_mm256_rcp_ps(x.raw()));
    }
    [[nodiscard]]
    inline vec8f rcp(const vec8i& x) {
        const __m256 r0 = _mm256_cvtepi32_ps(x.raw());
        return vec8f(_mm256_rcp_ps(r0));
    }
    [[nodiscard]]
    inline vec4d rcp(const vec4d& x) {
        return vec4d(set1(1.0) / x);
    }
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP