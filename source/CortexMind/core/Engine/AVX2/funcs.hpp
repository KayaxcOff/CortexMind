//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_FUNCS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_FUNCS_HPP

#include <CortexMind/core/Engine/AVX2/cmp.hpp>
#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief   Load 8 contiguous float values from aligned memory into a 256-bit vector
     * @param   src   Pointer to 32-byte aligned memory containing at least 8 floats
     * @return  AVX2 vector containing the loaded values
     */
    [[nodiscard]]
    inline vec8f load(const f32* src) {
        return _mm256_load_ps(src);
    }
    /**
     * @brief   Store 8 float values from a 256-bit vector to aligned memory
     * @param   dst   Pointer to 32-byte aligned memory where values will be stored
     * @param   src   Vector containing the 8 float values to store
     */
    inline void store(f32* dst, const vec8f src) {
        _mm256_store_ps(dst, src);
    }
    /**
     * @brief   Load 8 contiguous float values from unaligned memory
     * @param   src   Pointer to memory (alignment not required)
     * @return  AVX2 vector containing the loaded values
     */
    [[nodiscard]]
    inline vec8f loadu(const f32* src) {
        return _mm256_loadu_ps(src);
    }
    /**
     * @brief   Store 8 float values from a 256-bit vector to potentially unaligned memory
     * @param   dst   Pointer to destination memory (alignment not required)
     * @param   src   Vector containing the 8 float values to store
     */
    inline void storeu(f32* dst, const vec8f src) {
        _mm256_storeu_ps(dst, src);
    }

    /**
     * @brief   Element-wise addition of two 256-bit float vectors
     * @param   x   First input vector
     * @param   y   Second input vector
     * @return  Vector containing (x + y) for each lane
     */
    [[nodiscard]]
    inline vec8f add(const vec8f x, const vec8f y) {
        return _mm256_add_ps(x, y);
    }
    /**
     * @brief   Element-wise subtraction of two 256-bit float vectors
     * @param   x   Minuend vector
     * @param   y   Subtrahend vector
     * @return  Vector containing (x - y) for each lane
     */
    [[nodiscard]]
    inline vec8f sub(const vec8f x, const vec8f y) {
        return _mm256_sub_ps(x, y);
    }
    /**
     * @brief   Element-wise multiplication of two 256-bit float vectors
     * @param   x   First factor vector
     * @param   y   Second factor vector
     * @return  Vector containing (x × y) for each lane
     */
    [[nodiscard]]
    inline vec8f mul(const vec8f x, const vec8f y) {
        return _mm256_mul_ps(x, y);
    }
    /**
     * @brief   Element-wise division of two 256-bit float vectors
     * @param   x   Dividend vector
     * @param   y   Divisor vector (elements should be non-zero)
     * @return  Vector containing (x ÷ y) for each lane
     * @warning Division by zero produces undefined behavior (usually Inf or NaN)
     */
    [[nodiscard]]
    inline vec8f div(const vec8f x, const vec8f y) {
        return _mm256_div_ps(x, y);
    }

    /**
     * @brief   Broadcast a single float value to all 8 lanes of a 256-bit vector
     * @param   value   Scalar value to broadcast
     * @return  Vector with all elements equal to value
     */
    [[nodiscard]]
    inline vec8f set1(const f32 value) {
        return _mm256_set1_ps(value);
    }
    /**
     * @brief   Create a 256-bit vector with all elements set to 0.0f
     * @return  Zero-initialized AVX2 vector
     */
    [[nodiscard]]
    inline vec8f set_zero() {
        return _mm256_setzero_ps();
    }

    /**
     * @brief   Element-wise square root
     * @param   x   Input vector
     * @return  Vector containing √x for each lane
     * @note    Negative values produce NaN
     */
    [[nodiscard]]
    inline vec8f sqrt(const vec8f x) {
        return _mm256_sqrt_ps(x);
    }
    /**
     * @brief   Element-wise exponential function (e^x)
     * @param   x   Input vector
     * @return  Vector containing e^x for each lane
     */
    [[nodiscard]]
    inline vec8f exp(const vec8f x) {
        return _mm256_exp_ps(x);
    }
    /**
     * @brief   Element-wise natural logarithm
     * @param   x   Input vector (must be > 0)
     * @return  Vector containing ln(x) for each lane
     * @note    x ≤ 0 produces NaN or -Inf
     */
    [[nodiscard]]
    inline vec8f log(const vec8f x) {
        return _mm256_log_ps(x);
    }
    /**
     * @brief   Element-wise power function (x^y)
     * @param   x   Base vector
     * @param   y   Exponent vector
     * @return  Vector containing x^y for each lane
     * @note    Behavior for x ≤ 0 depends on y (may produce NaN)
     */
    [[nodiscard]]
    inline vec8f pow(const vec8f x, const vec8f y) {
        return _mm256_pow_ps(x, y);
    }
    /**
     * @brief   Fused multiply-add: x × y + z
     * @note    Uses _mm256_fmadd_ps (requires FMA extension)
     */
    [[nodiscard]]
    inline vec8f fma(const vec8f x, const vec8f y, const vec8f z) {
        return _mm256_fmadd_ps(x, y, z);
    }
    /**
     * @brief   Element-wise hyperbolic tangent (tanh)
     */
    [[nodiscard]]
    inline vec8f tanh(const vec8f x) {
        return _mm256_tanh_ps(x);
    }
    /**
     * @brief   Element-wise reciprocal square root
     * @note    Negative values produce NaN
     */
    [[nodiscard]]
    inline vec8f rsqrt(const vec8f x) {
        return _mm256_rsqrt_ps(x);
    }
    /**
     * @brief   Negate all elements: -x
     * @note    Uses XOR with sign bit mask
     */
    [[nodiscard]]
    inline vec8f neg(const vec8f x) {
        const vec8f sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<i32>(0x80000000u)));
        return _mm256_xor_ps(x, sign_mask);
    }
    /**
     * @brief   Fast reciprocal with Newton-Raphson refinement (1/x)
     * @note    More accurate than plain _mm256_rcp_ps
     */
    [[nodiscard]]
    inline vec8f rcp_nr(const vec8f& x) {
        const vec8f r0 = _mm256_rcp_ps(x);
        return _mm256_mul_ps(r0, _mm256_fnmadd_ps(x, r0, set1(2.0f)));
    }

    /**
     * @brief   Blend based on mask (blendv)
     * @note    mask lanes with MSB set select from y, otherwise x
     */
    [[nodiscard]]
    inline vec8f blendv(const vec8f& x, const vec8f& y, const vec8f& mask) {
        return _mm256_blendv_ps(x, y, mask);
    }
    /**
     * @brief   Element-wise minimum
     */
    [[nodiscard]]
    inline vec8f min(const vec8f& x, const vec8f& y) {
        return _mm256_min_ps(x, y);
    }
    /**
     * @brief   Element-wise maximum
     */
    [[nodiscard]]
    inline vec8f max(const vec8f& x, const vec8f& y) {
        return _mm256_max_ps(x, y);
    }

    /**
     * @brief   Horizontal sum of 8 floats
     * @return  Sum of all lanes
     */
    [[nodiscard]]
    inline f32 hsum(const vec8f& x) {
        const vec4f hi = _mm256_extractf128_ps(x, 1);
        const vec4f lo = _mm256_castps256_ps128(x);
        vec4f sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
    /**
     * @brief   Horizontal maximum
     */
    [[nodiscard]]
    inline f32 hmax(const vec8f& x) {
        const vec4f hi = _mm256_extractf128_ps(x, 1);
        const vec4f lo = _mm256_castps256_ps128(x);
        vec4f max = _mm_max_ps(lo, hi);
        max = _mm_max_ps(max, _mm_movehl_ps(max, max));
        max = _mm_max_ss(max, _mm_shuffle_ps(max, max, 1));
        return _mm_cvtss_f32(max);
    }
    /**
     * @brief   Horizontal minimum
     */
    [[nodiscard]]
    inline f32 hmin(const vec8f& x) {
        const vec4f hi = _mm256_extractf128_ps(x, 1);
        const vec4f lo = _mm256_castps256_ps128(x);
        vec4f min_val = _mm_min_ps(lo, hi);
        min_val = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
        min_val = _mm_min_ss(min_val, _mm_shuffle_ps(min_val, min_val, 1));
        return _mm_cvtss_f32(min_val);
    }

    /**
     * @brief   ReLU: max(x, 0)
     */
    [[nodiscard]]
    inline vec8f relu(const vec8f& x) {
        return max(set_zero(), x);
    }
    /**
     * @brief   Leaky ReLU: x > 0 ? x : alpha * x
     * @param   alpha   Negative slope (default 0.01)
     */
    [[nodiscard]]
    inline vec8f leaky_relu(const vec8f& x, const f32 alpha = 0.01f) {
        const vec8f mask = cmp::gt(x, set_zero());
        const vec8f neg_branch = mul(set1(alpha), x);
        return blendv(neg_branch, x, mask);
    }
    /**
     * @brief   Sigmoid: 1 / (1 + exp(-x))
     */
    [[nodiscard]]
    inline vec8f sigmoid(const vec8f& x) {
        const vec8f neg_x = neg(x);
        const vec8f exp_neg = exp(neg_x);
        const vec8f denom = add(set1(1.0f), exp_neg);
        return div(set1(1.0f), denom);
    }
    /**
     * @brief   Fast sigmoid using reciprocal
     */
    [[nodiscard]]
    inline vec8f sigmoid_fast(const vec8f& x) {
        const vec8f exp_neg = avx2::exp(neg(x));
        return rcp_nr(add(set1(1.0f), exp_neg));
    }
    /**
     * @brief   GELU (exact): 0.5 * x * (1 + erf(x / √2))
     * @note    Requires _mm256_erf_ps (SVML or compiler support)
     */
    [[nodiscard]]
    inline vec8f gelu_exact(const vec8f& x) {
        const vec8f inv_sqrt2 = set1(0.7071067811865475f);
        const vec8f erf_input = mul(x, inv_sqrt2);
        const vec8f erf_val = _mm256_erf_ps(erf_input);
        const vec8f one_plus_erf = add(set1(1.0f), erf_val);
        return mul(mul(set1(0.5f), x), one_plus_erf);
    }
    /**
     * @brief   GELU (approximate): 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     */
    [[nodiscard]]
    inline vec8f gelu(const vec8f& x) {
        const vec8f c0 = set1(0.7978845608028654f);
        const vec8f c1 = set1(0.044715f);

        const vec8f x3 = mul(mul(x, x), x);
        const vec8f inner = mul(c0, add(x, mul(c1, x3)));
        const vec8f tanh_val = tanh(inner);
        return mul(mul(set1(0.5f), x), add(set1(1.0f), tanh_val));
    }
    /**
     * @brief   SiLU: x * sigmoid(x)
     */
    [[nodiscard]]
    inline vec8f silu(const vec8f& x) {
        return mul(x, sigmoid(x));
    }
    /**
     * @brief   Swish (parameterized): x * sigmoid(beta * x)
     * @param   beta    Scaling factor (default 1.0)
     */
    [[nodiscard]]
    inline vec8f swish(const vec8f& x, const f32 beta = 1.0f) {
        return mul(x, sigmoid(mul(set1(beta), x)));
    }
    /**
     * @brief   Softmax over 8 elements: exp(x - max) / sum(exp(x - max))
     * @note    Single vector (8 elements) only — for batch use different kernel
     */
    [[nodiscard]]
    inline vec8f softmax(const vec8f& x) {
        const vec8f x_max = set1(hmax(x));
        const vec8f shifted = sub(x, x_max);
        const vec8f exp_val = exp(shifted);
        const vec8f sum = set1(hsum(exp_val));
        return div(exp_val, sum);
    }

    /**
     * @brief   Mean of 8 floats: hsum(x) / 8
     */
    [[nodiscard]]
    inline f32 mean(const vec8f x) {
        return hsum(x) * (1.0f / 8.0f);
    }
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_FUNCS_HPP