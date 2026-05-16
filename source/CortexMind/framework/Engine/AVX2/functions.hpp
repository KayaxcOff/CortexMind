//
// Created by muham on 7.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP

#include <CortexMind/framework/Engine/AVX2/cmp.hpp>
#include <CortexMind/framework/Engine/AVX2/fma.hpp>
#include <CortexMind/framework/Engine/AVX2/horizontal.hpp>
#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Loads 8 contiguous floats from aligned memory.
     * @param src Pointer to aligned memory (32-byte aligned).
     * @return __m256 vector containing the loaded values.
     * @note Requires 32-byte alignment. Use `loadu` for unaligned access.
     */
    [[nodiscard]]
    inline vec8f load(const f32* src) {
        return _mm256_load_ps(src);
    }
    /**
     * @brief Stores 8 floats to aligned memory.
     * @param dst Pointer to aligned destination memory.
     * @param src Source vector to store.
     * @note Requires 32-byte alignment.
     */
    inline void store(f32* dst, const vec8f src) {
        _mm256_store_ps(dst, src);
    }

    /**
     * @brief Loads 8 floats from unaligned memory.
     * @param src Pointer to memory (no alignment requirement).
     * @return vec8f vector.
     */
    [[nodiscard]]
    inline vec8f loadu(const f32* src) {
        return _mm256_loadu_ps(src);
    }
    /**
     * @brief Stores 8 floats to unaligned memory.
     * @param dst Destination pointer.
     * @param src Source vector.
     */
    inline void storeu(f32* dst, const vec8f src) {
        _mm256_storeu_ps(dst, src);
    }

    /**
     * @brief Broadcasts a single scalar value to all 8 lanes.
     * @param value Scalar value to broadcast.
     * @return vec8f with all elements equal to `value`.
     */
    [[nodiscard]]
    inline vec8f set1(const f32 value) {
        return _mm256_set1_ps(value);
    }
    /**
     * @brief Creates a vector with all elements set to zero.
     * @return Zeroed vec8f vector.
     */
    [[nodiscard]]
    inline vec8f zero() {
        return _mm256_setzero_ps();
    }

    /**
     * @brief Element-wise addition: `Xx + Xy`
     */
    [[nodiscard]]
    inline vec8f add(const vec8f Xx, const vec8f Xy) {
        return _mm256_add_ps(Xx, Xy);
    }
    /**
     * @brief Element-wise subtraction: `Xx - Xy`
     */
    [[nodiscard]]
    inline vec8f sub(const vec8f Xx, const vec8f Xy) {
        return _mm256_sub_ps(Xx, Xy);
    }
    /**
     * @brief Element-wise multiplication: `Xx * Xy`
     */
    [[nodiscard]]
    inline vec8f mul(const vec8f Xx, const vec8f Xy) {
        return _mm256_mul_ps(Xx, Xy);
    }
    /**
     * @brief Element-wise division: `Xx / Xy`
     */
    [[nodiscard]]
    inline vec8f div(const vec8f Xx, const vec8f Xy) {
        return _mm256_div_ps(Xx, Xy);
    }

    /**
     * @brief Element-wise negation (`-x`).
     * @note Uses XOR with sign a bit for best performance.
     */
    [[nodiscard]]
    inline vec8f neg(const vec8f x) {
        return _mm256_xor_ps(x, set1(-0.0f));
    }
    /**
     * @brief Element-wise exponential: e^x
     */
    [[nodiscard]]
    inline vec8f exp(const vec8f x) {
        return _mm256_exp_ps(x);
    }
    /**
     * @brief Element-wise base-2 exponential: 2^x
     */
    [[nodiscard]]
    inline vec8f exp2(const vec8f x) {
        return _mm256_exp2_ps(x);
    }
    /**
     * @brief Element-wise natural logarithm: ln(x)
     */
    [[nodiscard]]
    inline vec8f log(const vec8f x) {
        return _mm256_log_ps(x);
    }
    /**
     * @brief Element-wise base-2 logarithm: log2(x)
     */
    [[nodiscard]]
    inline vec8f log2(const vec8f x) {
        return _mm256_log2_ps(x);
    }
    /**
     * @brief Element-wise base-10 logarithm: log10(x)
     */
    [[nodiscard]]
    inline vec8f log10(const vec8f x) {
        return _mm256_log10_ps(x);
    }
    /**
     * @brief Element-wise square root.
     */
    [[nodiscard]]
    inline vec8f sqrt(const vec8f x) {
        return _mm256_sqrt_ps(x);
    }
    /**
     * @brief Element-wise reciprocal square root (1/sqrt(x)).
     */
    [[nodiscard]]
    inline vec8f rsqrt(const vec8f x) {
        return _mm256_rsqrt_ps(x);
    }
    /**
     * @brief Element-wise sine.
     */
    [[nodiscard]]
    inline vec8f sin(const vec8f x) {
        return _mm256_sin_ps(x);
    }
    /**
     * @brief Element-wise cos.
     */
    [[nodiscard]]
    inline vec8f cos(const vec8f x) {
        return _mm256_cos_ps(x);
    }
    /**
     * @brief Element-wise hyperbolic sine.
     */
    [[nodiscard]]
    inline vec8f sinh(const vec8f x) {
        return _mm256_sinh_ps(x);
    }
    /**
     * @brief Element-wise tangent.
     */
    [[nodiscard]]
    inline vec8f tan(const vec8f x) {
        return _mm256_tan_ps(x);
    }
    /**
     * @brief Element-wise hyperbolic tangent.
     */
    [[nodiscard]]
    inline vec8f tanh(const vec8f x) {
        return _mm256_tanh_ps(x);
    }
    /**
     * @brief Element-wise power: Xx^Xy
     */
    [[nodiscard]]
    inline vec8f pow(const vec8f Xx, const vec8f Xy) {
        return _mm256_pow_ps(Xx, Xy);
    }
    /**
     * @brief Element-wise absolute value.
     */
    [[nodiscard]]
    inline vec8f abs(const vec8f x) {
        return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    }
    /**
     * @brief Newton-Raphson reciprocal approximation (high precision).
     * @note Performs one Newton-Raphson iteration for better accuracy.
     */
    [[nodiscard]]
    inline vec8f rcp_nr(const vec8f x) {
        const vec8f r0 = _mm256_rcp_ps(x);
        return mul(r0, fma::nadd(x, r0, set1(2.0f)));
    }
    /**
     * @brief Shuffles elements according to immediate control.
     * @tparam imm Shuffle control constant (0-255).
     */
    template<i32 imm>
    [[nodiscard]]
    vec8f shuffle(const vec8f Xx, const vec8f Xy) {
        return _mm256_shuffle_ps(Xx, Xy, imm);
    }

    /**
     * @brief Blend two vectors using a mask.
     * @param Xx Source vector when mask bit = 0
     * @param Xy Source vector when mask bit = 1
     * @param Xz Mask vector (sign bit controls selection)
     */
    [[nodiscard]]
    inline vec8f blendv(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
        return _mm256_blendv_ps(Xx, Xy, Xz);
    }
    /**
     * @brief Element-wise maximum.
     */
    [[nodiscard]]
    inline vec8f max(const vec8f Xx, const vec8f Xy) {
        return _mm256_max_ps(Xx, Xy);
    }
    /**
     * @brief Element-wise minimum.
     */
    [[nodiscard]]
    inline vec8f min(const vec8f Xx, const vec8f Xy) {
        return _mm256_min_ps(Xx, Xy);
    }

    /**
     * @brief ReLU activation: max(0, x)
     */
    [[nodiscard]]
    inline vec8f relu(const vec8f x) {
        return max(zero(), x);
    }
    /**
     * @brief Leaky ReLU activation.
     * @param x Input vector
     * @param alpha Negative slope coefficient (default 0.01)
     */
    [[nodiscard]]
    inline vec8f leaky_relu(const vec8f x, const f32 alpha = 0.01f) {
        const vec8f mask = cmp::gt(x, zero());
        const vec8f neg_branch = mul(set1(alpha), x);
        return blendv(neg_branch, x, mask);
    }
    /**
     * @brief Sigmoid activation: 1 / (1 + e^(-x))
     */
    [[nodiscard]]
    inline vec8f sigmoid(const vec8f x) {
        const vec8f neg_x = neg(x);
        const vec8f exp_neg = exp(neg_x);
        const vec8f denom = add(set1(1.0f), exp_neg);
        return div(set1(1.0f), denom);
    }
    /**
     * @brief Fast sigmoid approximation using reciprocal.
     */
    [[nodiscard]]
    inline vec8f sigmoid_fast(const vec8f x) {
        const vec8f exp_neg = exp(neg(x));
        return rcp_nr(add(set1(1.0f), exp_neg));
    }
    /**
     * @brief Exact GELU using erf function.
     */
    [[nodiscard]]
    inline vec8f gelu_exact(const vec8f x) {
        const vec8f inv_sqrt2 = set1(0.7071067811865475f);
        const vec8f erf_input = mul(x, inv_sqrt2);
        const vec8f erf_val = _mm256_erf_ps(erf_input);
        const vec8f one_plus_erf = add(set1(1.0f), erf_val);
        return mul(mul(set1(0.5f), x), one_plus_erf);
    }
    /**
     * @brief Approximate GELU using tanh (most common implementation).
     */
    [[nodiscard]]
    inline vec8f gelu(const vec8f x) {
        const vec8f c0 = set1(0.7978845608028654f);
        const vec8f c1 = set1(0.044715f);

        const vec8f x3 = mul(mul(x, x), x);
        const vec8f inner = mul(c0, add(x, mul(c1, x3)));
        const vec8f tanh_val = tanh(inner);
        return mul(mul(set1(0.5f), x), add(set1(1.0f), tanh_val));
    }
    /**
     * @brief SiLU (Sigmoid Linear Unit) activation.
     */
    [[nodiscard]]
    inline vec8f silu(const vec8f x) {
        return mul(x, sigmoid(x));
    }
    /**
     * @brief Swish activation with configurable beta.
     * @param beta Scaling factor for the sigmoid (default = 1.0)
     */
    [[nodiscard]]
    inline vec8f swish(const vec8f x, const f32 beta = 1.0f) {
        return mul(x, sigmoid(mul(set1(beta), x)));
    }
    /**
     * @brief Softmax along the vector (single vector version).
     */
    [[nodiscard]]
    inline vec8f softmax(const vec8f x) {
        const vec8f x_max = set1(horizontal::max(x));
        const vec8f shifted = sub(x, x_max);
        const vec8f exp_val = exp(shifted);
        const vec8f sum = set1(horizontal::sum(exp_val));
        return div(exp_val, sum);
    }
    /**
     * @brief Fast SiLU using sigmoid_fast.
     */
    [[nodiscard]]
    inline vec8f silu_fast(const vec8f x) {
        return mul(x, sigmoid_fast(x));
    }
    /**
     * @brief Fast Swish using sigmoid_fast.
     * @param beta Scaling factor (default = 1.0)
     */
    [[nodiscard]]
    inline vec8f swish_fast(const vec8f x, const f32 beta = 1.0f) {
        return mul(x, sigmoid_fast(mul(set1(beta), x)));
    }
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FUNCTIONS_HPP