//
// Created by muham on 21.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_FUNCS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_FUNCS_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    [[nodiscard]] inline
    vec8f load(const f32* src) {
        return _mm256_load_ps(src);
    }
    inline
    void store(f32 *src, const vec8f &dst) {
        _mm256_store_ps(src, dst);
    }
    [[nodiscard]] inline
    vec8f loadu(const f32* src) {
        return _mm256_loadu_ps(src);
    }
    inline
    void storeu(f32 *src, const vec8f &dst) {
        _mm256_storeu_ps(src, dst);
    }

    [[nodiscard]] inline
    vec8f set1(const f32 value) {
        return _mm256_set1_ps(value);
    }
    [[nodiscard]] inline
    vec8f zero() {
        return _mm256_setzero_ps();
    }

    [[nodiscard]] inline
    vec8f add(const vec8f &x, const vec8f &y) {
        return _mm256_add_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f sub(const vec8f &x, const vec8f &y) {
        return _mm256_sub_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f mul(const vec8f &x, const vec8f &y) {
        return _mm256_mul_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f div(const vec8f &x, const vec8f &y) {
        return _mm256_div_ps(x, y);
    }

    [[nodiscard]] inline
    vec8f sqrt(const vec8f &x) {
        return _mm256_sqrt_ps(x);
    }
    [[nodiscard]] inline
    vec8f exp(const vec8f &x) {
        return _mm256_exp_ps(x);
    }
    [[nodiscard]] inline
    vec8f log(const vec8f &x) {
        return _mm256_log_ps(x);
    }
    [[nodiscard]] inline
    vec8f log2(const vec8f &x) {
        return _mm256_log2_ps(x);
    }
    [[nodiscard]] inline
    vec8f log10(const vec8f &x) {
        return _mm256_log10_ps(x);
    }
    [[nodiscard]] inline
    vec8f exp2(const vec8f &x) {
        return _mm256_exp2_ps(x);
    }
    [[nodiscard]] inline
    vec8f exp10(const vec8f &x) {
        return _mm256_exp10_ps(x);
    }
    [[nodiscard]] inline
    vec8f pow(const vec8f &x, const vec8f &y) {
        return _mm256_pow_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f sin(const vec8f &x) {
        return _mm256_sin_ps(x);
    }
    [[nodiscard]] inline
    vec8f sinh(const vec8f &x) {
        return _mm256_sinh_ps(x);
    }
    [[nodiscard]] inline
    vec8f cos(const vec8f &x) {
        return _mm256_cos_ps(x);
    }
    [[nodiscard]] inline
    vec8f cosh(const vec8f &x) {
        return _mm256_cosh_ps(x);
    }
    [[nodiscard]] inline
    vec8f tan(const vec8f &x) {
        return _mm256_tan_ps(x);
    }
    [[nodiscard]] inline
    vec8f tanh(const vec8f &x) {
        return _mm256_tanh_ps(x);
    }
    [[nodiscard]] inline
    vec8f abs(const vec8f _x) {
        const vec8f mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return _mm256_and_ps(_x, mask);
    }
    [[nodiscard]] inline
    vec8f fma(const vec8f &x, const vec8f &y, const vec8f &z) {
        return _mm256_fmadd_ps(x, y, z);
    }
    [[nodiscard]] inline
    vec8f rsqrt(const vec8f &x) {
        return _mm256_rsqrt_ps(x);
    }

    [[nodiscard]] inline
    vec8f blendv(const vec8f& x, const vec8f& y, const vec8f& mask) {
        return _mm256_blendv_ps(x, y, mask);
    }
    [[nodiscard]] inline
    vec8f min(const vec8f& x, const vec8f& y) {
        return _mm256_min_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f max(const vec8f& x, const vec8f& y) {
        return _mm256_max_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f and_(const vec8f& x, const vec8f& y) {
        return _mm256_and_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f or_(const vec8f& x, const vec8f& y) {
        return _mm256_or_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f xor_(const vec8f& x, const vec8f& y) {
        return _mm256_xor_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f andnot_(const vec8f& x, const vec8f& y) {
        return _mm256_andnot_ps(x, y);
    }
    [[nodiscard]] inline
    vec8f clamp(const vec8f& x, const vec8f& lo, const vec8f& hi) {
        return max(lo, min(x, hi));
    }
    [[nodiscard]] inline
    vec8f clamp(const vec8f& x, const f32 lo, const f32 hi) {
        return clamp(x, set1(lo), set1(hi));
    }

    template<i32 dest>
    [[nodiscard]]
    vec8f permute(const vec8f& x) {
        return _mm256_permute_ps(x, dest);
    }
    template<i32 dest>
    [[nodiscard]]
    vec8f shuffle(const vec8f& x, const vec8f& y) {
        return _mm256_shuffle_ps(x, y, dest);
    }

    struct cmp {
        [[nodiscard]]
        static vec8f gt(const vec8f& x, const vec8f& y) {
            return _mm256_cmp_ps(x, y, _CMP_GT_OS);
        }
        [[nodiscard]]
        static vec8f lt(const vec8f& x, const vec8f& y) {
            return _mm256_cmp_ps(x, y, _CMP_LT_OS);
        }
        [[nodiscard]]
        static vec8f eq(const vec8f& x, const vec8f& y) {
            return _mm256_cmp_ps(x, y, _CMP_EQ_OS);
        }
        [[nodiscard]]
        static vec8f ge(const vec8f& x, const vec8f& y) {
            return _mm256_cmp_ps(x, y, _CMP_GE_OS);
        }
        [[nodiscard]]
        static vec8f le(const vec8f& x, const vec8f& y) {
            return _mm256_cmp_ps(x, y, _CMP_LE_OS);
        }
        [[nodiscard]]
        static vec8f ne(const vec8f& x, const vec8f& y) {
            return _mm256_cmp_ps(x, y, _CMP_NEQ_OS);
        }
        [[nodiscard]]
        static int mask(const vec8f& x) {
            return _mm256_movemask_ps(x);
        }
        [[nodiscard]]
        static bool any(const vec8f& x) {
            return _mm256_movemask_ps(x) != 0;
        }
        [[nodiscard]]
        static bool all(const vec8f& x) {
            return _mm256_movemask_ps(x) == 0xFF;
        }
    };

    [[nodiscard]] inline
    f32 hsum(const vec8f& x) {
        const vec4f hi = _mm256_extractf128_ps(x, 1);
        const vec4f lo = _mm256_castps256_ps128(x);
        vec4f sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
    [[nodiscard]] inline
    f32 hmax(const vec8f& x) {
        const vec4f hi = _mm256_extractf128_ps(x, 1);
        const vec4f lo = _mm256_castps256_ps128(x);
        vec4f max = _mm_max_ps(lo, hi);
        max = _mm_max_ps(max, _mm_movehl_ps(max, max));
        max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
        return _mm_cvtss_f32(max);
    }
    [[nodiscard]] inline
    f32 hmin(const vec8f& x) {
        const vec4f hi = _mm256_extractf128_ps(x, 1);
        const vec4f lo = _mm256_castps256_ps128(x);
        vec4f min_val = _mm_min_ps(lo, hi);
        min_val = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
        min_val = _mm_min_ss(min_val, _mm_shuffle_ps(min_val, min_val, 1));
        return _mm_cvtss_f32(min_val);
    }

    [[nodiscard]] inline
    vec8f neg(const vec8f& x) {
        const vec8f sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
        return _mm256_xor_ps(x, sign_mask);
    }
    [[nodiscard]] inline
    vec8f sign(const vec8f& x) {
        const vec8f pos = _mm256_and_ps(cmp::gt(x, zero()), set1(1.0f));
        const vec8f neg_ = _mm256_and_ps(cmp::lt(x, zero()), set1(-1.0f));
        return _mm256_or_ps(pos, neg_);
    }
    [[nodiscard]] inline
    vec8f rcp_nr(const vec8f& x) {
        const vec8f r0 = _mm256_rcp_ps(x);
        return _mm256_mul_ps(r0, _mm256_fnmadd_ps(x, r0, set1(2.0f)));
    }

    [[nodiscard]] inline
    vec8f relu(const vec8f& x) {
        return max(zero(), x);
    }
    [[nodiscard]] inline
    vec8f leaky_relu(const vec8f& x, const f32 alpha = 0.01f) {
        const vec8f mask = cmp::gt(x, zero());
        const vec8f neg_branch = mul(set1(alpha), x);
        return blendv(neg_branch, x, mask);
    }
    [[nodiscard]] inline
    vec8f sigmoid(const vec8f& x) {
        const vec8f neg_x = neg(x);
        const vec8f exp_neg = avx2::exp(neg_x);
        const vec8f denom = add(set1(1.0f), exp_neg);
        return div(set1(1.0f), denom);
    }
    [[nodiscard]] inline
    vec8f sigmoid_fast(const vec8f& x) {
        const vec8f exp_neg = avx2::exp(neg(x));
        return rcp_nr(add(set1(1.0f), exp_neg));
    }
    [[nodiscard]] inline
    vec8f gelu_exact(const vec8f& x) {
        const vec8f inv_sqrt2 = set1(0.7071067811865475f);
        const vec8f erf_input = mul(x, inv_sqrt2);
        const vec8f erf_val = _mm256_erf_ps(erf_input);
        const vec8f one_plus_erf = add(set1(1.0f), erf_val);
        return mul(mul(set1(0.5f), x), one_plus_erf);
    }
    [[nodiscard]] inline
    vec8f gelu(const vec8f& x) {
        const vec8f c0 = set1(0.7978845608028654f);
        const vec8f c1 = set1(0.044715f);

        const vec8f x3 = mul(mul(x, x), x);
        const vec8f inner = mul(c0, add(x, mul(c1, x3)));
        const vec8f tanh_val = tanh(inner);
        return mul(mul(set1(0.5f), x), add(set1(1.0f), tanh_val));
    }
    [[nodiscard]] inline
    vec8f silu(const vec8f& x) {
        return mul(x, sigmoid(x));
    }
    [[nodiscard]] inline
    vec8f swish(const vec8f& x, const f32 beta = 1.0f) {
        return mul(x, sigmoid(mul(set1(beta), x)));
    }
    [[nodiscard]] inline
    vec8f softmax(const vec8f& x) {
        const vec8f x_max = set1(hmax(x));
        const vec8f shifted = sub(x, x_max);
        const vec8f exp_val = avx2::exp(shifted);
        const vec8f sum = set1(hsum(exp_val));
        return div(exp_val, sum);
    }

    [[nodiscard]] inline
    f32 mean(const vec8f x) {
        return hsum(x) * (1.0f / 8.0f);
    }
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_FUNCS_HPP