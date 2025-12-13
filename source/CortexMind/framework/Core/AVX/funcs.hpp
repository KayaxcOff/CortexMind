//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_FUNCS_HPP
#define CORTEXMIND_FUNCS_HPP

#include <CortexMind/framework/Core/AVX/variable.hpp>

namespace cortex::_fw::avx2 {
    inline reg load(const float* ptr)                {return _mm256_loadu_ps(ptr);}
    inline void store(float* ptr, const reg value)   {_mm256_storeu_ps(ptr, value);}
    inline reg load_u(const float* ptr)              {return _mm256_loadu_ps(ptr);}
    inline void store_u(float* ptr, const reg value) {_mm256_storeu_ps(ptr, value);}

    inline reg load_partial(const float* ptr, const size_t idx) {
        if (idx == 8) return load(ptr);
        int mask_vals[8] = {};
        for (size_t i = 0; i < idx && i < 8; ++i) mask_vals[i] = -1;
        const regi mask = _mm256_loadu_si256(reinterpret_cast<const regi*>(mask_vals));
        return _mm256_maskload_ps(ptr, mask);
    }
    inline void store_partial(float* ptr, const reg value, const size_t idx) {
        if (idx == 8) { store(ptr, value); return; }
        int mask_vals[8] = {};
        for (size_t i = 0; i < idx && i < 8; ++i) mask_vals[i] = -1;
        const regi mask = _mm256_loadu_si256(reinterpret_cast<const regi*>(mask_vals));
        _mm256_maskstore_ps(ptr, mask, value);
    }

    inline reg add(const reg a, const reg b) noexcept               { return _mm256_add_ps(a, b); }
    inline reg sub(const reg a, const reg b) noexcept               { return _mm256_sub_ps(a, b); }
    inline reg mul(const reg a, const reg b) noexcept               { return _mm256_mul_ps(a, b); }
    inline reg div(const reg a, const reg b) noexcept               { return _mm256_div_ps(a, b); }
    inline reg fma(const reg a, const reg b, const reg c) noexcept  { return _mm256_fmadd_ps(a, b, c); }
    inline reg neg(const reg a) noexcept                            { return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f)); }
    inline reg abs(const reg a) noexcept                            { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
    inline reg broadcast(const float x) noexcept                    { return _mm256_set1_ps(x); }
    inline reg zero() noexcept                                      { return _mm256_setzero_ps(); }
    inline reg sqrt(const reg x) noexcept                           { return _mm256_sqrt_ps(x); }
    inline reg r_sqrt(const reg x) noexcept                         { return _mm256_rsqrt_ps(x); }

    inline reg exp_approx(reg x) noexcept {
        const reg max_x = _mm256_set1_ps(88.722839f);
        const reg min_x = _mm256_set1_ps(-103.972084f);
        x = _mm256_min_ps(max_x, _mm256_max_ps(min_x, x));

        const reg ln2     = _mm256_set1_ps(0.6931471805599453f);
        const reg inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
        const reg y = _mm256_mul_ps(x, inv_ln2);

        const __m256i yi = _mm256_cvttps_epi32(y);
        const reg y_floor = _mm256_cvtepi32_ps(yi);

        const reg r = _mm256_sub_ps(x, _mm256_mul_ps(y_floor, ln2));

        const reg c5 = _mm256_set1_ps(1.0f / 120.0f);
        const reg c4 = _mm256_set1_ps(1.0f / 24.0f);
        const reg c3 = _mm256_set1_ps(1.0f / 6.0f);
        const reg c2 = _mm256_set1_ps(0.5f);
        const reg c1 = _mm256_set1_ps(1.0f);

        reg poly = _mm256_fmadd_ps(c5, r, c4);
        poly = _mm256_fmadd_ps(poly, r, c3);
        poly = _mm256_fmadd_ps(poly, r, c2);
        poly = _mm256_fmadd_ps(poly, r, c1);

        const __m256i bias = _mm256_set1_epi32(127);
        __m256i ei = _mm256_add_epi32(yi, bias);
        ei = _mm256_slli_epi32(ei, 23);
        const reg pow2n = _mm256_castsi256_ps(ei);

        return _mm256_mul_ps(poly, pow2n);
    }

    inline reg log_approx(const reg x) noexcept {
        const reg zero = _mm256_set1_ps(0.0f);
        const reg ln2  = _mm256_set1_ps(0.6931471805599453f);

        const __m256i ix = _mm256_castps_si256(x);
        const __m256i bias = _mm256_set1_epi32(127);
        __m256i exp_i = _mm256_srli_epi32(ix, 23);
        exp_i = _mm256_sub_epi32(exp_i, bias);
        const reg e = _mm256_cvtepi32_ps(exp_i);

        const __m256i mask = _mm256_set1_epi32(0x007FFFFF);
        const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
        const __m256i mant_i = _mm256_or_si256(_mm256_and_si256(ix, mask), one_bits);
        const reg m = _mm256_castsi256_ps(mant_i);

        const reg r = _mm256_div_ps(_mm256_sub_ps(m, _mm256_set1_ps(1.0f)), _mm256_add_ps(m, _mm256_set1_ps(1.0f)));

        const reg r2 = _mm256_mul_ps(r, r);

        const reg c3 = _mm256_set1_ps(1.0f / 3.0f);
        const reg c1 = _mm256_set1_ps(1.0f);

        reg poly = _mm256_fmadd_ps(c3, r2, c1);
        poly = _mm256_mul_ps(poly, r);

        reg result = _mm256_fmadd_ps(e, ln2, poly);

        const reg nanv = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
        const reg neginf = _mm256_set1_ps(-INFINITY);
        result = _mm256_blendv_ps(result, nanv, _mm256_cmp_ps(x, zero, _CMP_LT_OQ));
        result = _mm256_blendv_ps(result, neginf, _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));

        return result;
    }

    inline float h_sum(const reg v) noexcept {
        const __m128 lo = _mm256_castps256_ps128(v);
        const __m128 hi = _mm256_extractf128_ps(v, 1);

        __m128 s = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(s);
        s = _mm_add_ps(s, shuf);
        shuf = _mm_movehl_ps(shuf, s);
        s = _mm_add_ss(s, shuf);

        return _mm_cvtss_f32(s);
    }

    inline float max(const reg v) noexcept {
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        const __m128 lo = _mm256_castps256_ps128(v);
        __m128 m = _mm_max_ps(hi, lo);
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,3,2)));
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,0,1)));
        return _mm_cvtss_f32(m);
    }

    inline float min(const reg v) noexcept {
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        const __m128 lo = _mm256_castps256_ps128(v);
        __m128 m = _mm_min_ps(hi, lo);
        m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,3,2)));
        m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,0,0,1)));
        return _mm_cvtss_f32(m);
    }
}

#endif //CORTEXMIND_FUNCS_HPP