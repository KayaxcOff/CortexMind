//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP

#include <CortexMind/framework/Engine/AVX2/fma.hpp>
#include <cstdint>

namespace cortex::_fw::avx2 {
    [[nodiscard]]
    inline vec8f load(const float* src) {
        return _mm256_load_ps(src);
    }
    [[nodiscard]]
    inline vec8i load(const std::int32_t* src) {
        return _mm256_load_si256(reinterpret_cast<const vec8i*>(src));
    }
    [[nodiscard]]
    inline vec4d load(const double* src) {
        return _mm256_load_pd(src);
    }

    inline void store(float* dst, const vec8f& src) {
        _mm256_store_ps(dst, src);
    }
    inline void store(std::int32_t* dst, const vec8i& src) {
        _mm256_store_si256(reinterpret_cast<vec8i*>(dst), src);
    }
    inline void store(double* dst, const vec4d& src) {
        _mm256_store_pd(dst, src);
    }


    [[nodiscard]]
    inline vec8f loadu(const float* src) {
        return _mm256_loadu_ps(src);
    }
    [[nodiscard]]
    inline vec8i loadu(const std::int32_t* src) {
        return _mm256_load_si256(reinterpret_cast<const vec8i*>(src));
    }
    inline vec4d loadu(const double* src) {
        return _mm256_loadu_pd(src);
    }

    inline void storeu(float* dst, const vec8f& src) {
        _mm256_storeu_ps(dst, src);
    }
    inline void storeu(std::int32_t* dst, const vec8i& src) {
        _mm256_store_si256(reinterpret_cast<vec8i*>(dst), src);
    }
    inline void storeu(double* dst, const vec4d& src) {
        _mm256_store_pd(dst, src);
    }


    [[nodiscard]]
    inline vec8f set1(const float value) {
        return _mm256_set1_ps(value);
    }
    [[nodiscard]]
    inline vec8i set1(const std::int32_t value) {
        return _mm256_set1_epi32(value);
    }
    [[nodiscard]]
    inline vec4d set1(const double value) {
        return _mm256_set1_pd(value);
    }

    [[nodiscard]]
    inline vec8f zerof() {
        return _mm256_setzero_ps();
    }
    [[nodiscard]]
    inline vec8i zeroi() {
        return _mm256_setzero_si256();
    }
    [[nodiscard]]
    inline vec4d zerod() {
        return _mm256_setzero_pd();
    }


    [[nodiscard]]
    inline vec8f add(const vec8f& x1, const vec8f& x2) {
        return _mm256_add_ps(x1, x2);
    }
    [[nodiscard]]
    inline vec8i add(const vec8i& x1, const vec8i& x2) {
        return _mm256_add_epi32(x1, x2);
    }
    [[nodiscard]]
    inline vec4d add(const vec4d& x1, const vec4d& x2) {
        return _mm256_add_pd(x1, x2);
    }

    [[nodiscard]]
    inline vec8f sub(const vec8f& x1, const vec8f& x2) {
        return _mm256_sub_ps(x1, x2);
    }
    [[nodiscard]]
    inline vec8i sub(const vec8i& x1, const vec8i& x2) {
        return _mm256_sub_epi32(x1, x2);
    }
    [[nodiscard]]
    inline vec4d sub(const vec4d& x1, const vec4d& x2) {
        return _mm256_sub_pd(x1, x2);
    }

    [[nodiscard]]
    inline vec8f mul(const vec8f& x1, const vec8f& x2) {
        return _mm256_mul_ps(x1, x2);
    }
    [[nodiscard]]
    inline vec8i mul(const vec8i& x1, const vec8i& x2) {
        return _mm256_mullo_epi32(x1, x2);
    }
    [[nodiscard]]
    inline vec4d mul(const vec4d& x1, const vec4d& x2) {
        return _mm256_mul_pd(x1, x2);
    }

    [[nodiscard]]
    inline vec8f div(const vec8f& x1, const vec8f& x2) {
        return _mm256_div_ps(x1, x2);
    }
    [[nodiscard]]
    inline vec8i div(const vec8i& x1, const vec8i& x2) {
        return _mm256_div_epi32(x1, x2);
    }
    [[nodiscard]]
    inline vec4d div(const vec4d& x1, const vec4d& x2) {
        return _mm256_div_pd(x1, x2);
    }


    [[nodiscard]]
    inline vec8f square(const vec8f& x) {
        return mul(x, x);
    }
    [[nodiscard]]
    inline vec4d square(const vec4d& x) {
        return mul(x, x);
    }

    [[nodiscard]]
    inline vec8f pow(const vec8f& x1, const vec8f& x2) {
        return _mm256_pow_ps(x1, x2);
    }
    [[nodiscard]]
    inline vec4d pow(const vec4d& x1, const vec4d& x2) {
        return _mm256_pow_pd(x1, x2);
    }

    [[nodiscard]]
    inline vec8f sqrt(const vec8f& x) {
        return _mm256_sqrt_ps(x);
    }
    [[nodiscard]]
    inline vec4d sqrt(const vec4d& x) {
        return _mm256_sqrt_pd(x);
    }

    [[nodiscard]]
    inline vec8f rsqrt(const vec8f& x) {
        return _mm256_rsqrt_ps(x);
    }
    [[nodiscard]]
    inline vec4d rsqrt(const vec4d& x) {
        return mul(set1(1.0), sqrt(x));
    }

    [[nodiscard]]
    inline vec8f log(const vec8f& x) {
        return _mm256_log_ps(x);
    }
    [[nodiscard]]
    inline vec4d log(const vec4d& x) {
        return _mm256_log_pd(x);
    }

    [[nodiscard]]
    inline vec8f exp(const vec8f& x) {
        return _mm256_exp_ps(x);
    }
    [[nodiscard]]
    inline vec4d exp(const vec4d& x) {
        return _mm256_exp_pd(x);
    }

    [[nodiscard]]
    inline vec8f erf(const vec8f& x) {
        return _mm256_erf_ps(x);
    }
    [[nodiscard]]
    inline vec4d erf(const vec4d& x) {
        return _mm256_erf_pd(x);
    }

    [[nodiscard]]
    inline vec8f sin(const vec8f& x) {
        return _mm256_sin_ps(x);
    }
    [[nodiscard]]
    inline vec4d sin(const vec4d& x) {
        return _mm256_sin_pd(x);
    }

    [[nodiscard]]
    inline vec8f cos(const vec8f& x) {
        return _mm256_cos_ps(x);
    }
    [[nodiscard]]
    inline vec4d cos(const vec4d& x) {
        return _mm256_cos_pd(x);
    }

    [[nodiscard]]
    inline vec8f abs(const vec8f& x) {
        return _mm256_andnot_ps(set1(-0.0f), x);
    }
    [[nodiscard]]
    inline vec4d abs(const vec4d& x) {
        return _mm256_andnot_pd(set1(-0.0), x);
    }

    [[nodiscard]]
    inline vec8f neg(const vec8f& x) {
        return _mm256_xor_ps(x, set1(-1.0f));
    }
    [[nodiscard]]
    inline vec4d neg(const vec4d& x) {
        return _mm256_xor_pd(x, set1(-1.0));
    }

    [[nodiscard]]
    inline vec8f rcp(const vec8f& x) {
        return _mm256_rcp_ps(x);
    }
    [[nodiscard]]
    inline vec4d rcp(const vec4d& x) {
        return div(set1(1.0), x);
    }

    [[nodiscard]]
    inline vec8f nr(const vec8f& x) {
        const vec8f r0 = rcp(x);
        return mul(r0, fma::nadd(x, r0, set1(2.0f)));
    }
    [[nodiscard]]
    inline vec4d nr(const vec4d& x) {
        const vec4d r0 = rcp(x);
        return mul(r0, fma::nadd(x, r0, set1(2.0)));
    }


    template<std::int32_t imm>
    [[nodiscard]]
    vec8f shuffle(const vec8f& x1, const vec8f& x2) {
        return _mm256_shuffle_ps(x1, x2, imm);
    }
    template<std::int32_t imm>
    [[nodiscard]]
    vec4d shuffle(const vec4d& x1, const vec4d& x2) {
        return _mm256_shuffle_pd(x1, x2, imm);
    }

    [[nodiscard]]
    inline vec8f blendv(const vec8f& x1, const vec8f& x2, const vec8f& x3) {
        return _mm256_blendv_ps(x1, x2, x3);
    }
    [[nodiscard]]
    inline vec4d blendv(const vec4d& x1, const vec4d& x2, const vec4d& x3) {
        return _mm256_blendv_pd(x1, x2, x3);
    }

    [[nodiscard]]
    inline vec8f max(const vec8f& x1, const vec8f& x2) {
        return _mm256_max_ps(x1, x2);
    }
    [[nodiscard]]
    inline vec4d max(const vec4d& x1, const vec4d& x2) {
        return _mm256_max_pd(x1, x2);
    }

    [[nodiscard]]
    inline vec8f min(const vec8f& x1, const vec8f& x2) {
        return _mm256_min_ps(x1, x2);
    }
    [[nodiscard]]
    inline vec4d min(const vec4d& x1, const vec4d& x2) {
        return _mm256_min_pd(x1, x2);
    }

    [[nodiscard]]
    inline vec8f relu(const vec8f& x) {
        return max(zerof(), x);
    }
    [[nodiscard]]
    inline vec4d relu(const vec4d& x) {
        return max(zerod(), x);
    }

    [[nodiscard]]
    inline vec8f tanh(const vec8f& x) {
        return _mm256_tanh_ps(x);
    }
    [[nodiscard]]
    inline vec4d tanh(const vec4d& x) {
        return _mm256_tanh_pd(x);
    }
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP