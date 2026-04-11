//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_FUNCTIONS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_FUNCTIONS_HPP

#include <CortexMind/core/Engine/AVX2/cmp.hpp>
#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/core/Engine/AVX2/horizontal.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Loads 8 packed 32-bit floats from memory into
     * a __m256 vector.
     * @warning The float pointer must be 32-byte aligned.
     * @return __m256 containing 8 packed 32-bit floats.
     */
    [[nodiscard]]
    inline vec8f load(const f32* src) {
        return _mm256_load_ps(src);
    }
    /**
     * @brief Stores 8 packed 32-bit floats from a __m256
     * vector to memory.
     * @warning The destination pointer must be 32-byte aligned.
     */
    inline void store(f32* dst, const vec8f src) {
        _mm256_store_ps(dst, src);
    }

    /**
     * @brief Loads 8 packed 32-bit floats from memory into
     * a __m256 vector.
     * @note 32-bit floats do not necessarily have
     * to be aligned.
     * @return __m256 containing 8 packed 32-bit floats.
     */
    [[nodiscard]]
    inline vec8f loadu(const f32* src) {
        return _mm256_loadu_ps(src);
    }
    /**
     * @brief Stores 8 packed 32-bit floats from a __m256 vector to memory.
     * @note The destination pointer does not need to be 32-byte aligned.
     */
    inline void storeu(f32* dst, const vec8f src) {
        _mm256_storeu_ps(dst, src);
    }

    /**
     * @brief Creates a __m256 vector with all elements set to a given value.
     * @param value The 32-bit float value to set in all elements.
     * @return __m256 vector with all elements set to the given value.
     */
    [[nodiscard]]
    inline vec8f set1(const f32 value) {
        return _mm256_set1_ps(value);
    }
    /**
     * @brief Creates a __m256 vector with all elements set to zero.
     * @return __m256 vector with all elements set to zero.
     */
    [[nodiscard]]
    inline vec8f zero() {
        return _mm256_setzero_ps();
    }

    /**
     * @brief Adds two __m256 vectors element-wise.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @return __m256 vector where each element is Xx[i] + Xy[i].
     */
    [[nodiscard]]
    inline vec8f add(const vec8f Xx, const vec8f Xy) {
        return _mm256_add_ps(Xx, Xy);
    }
    /**
     * @brief Subtracts two __m256 vectors element-wise.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @return __m256 vector where each element is Xx[i] - Xy[i].
     */
    [[nodiscard]]
    inline vec8f sub(const vec8f Xx, const vec8f Xy) {
        return _mm256_sub_ps(Xx, Xy);
    }
    /**
     * @brief Multiplies two __m256 vectors element-wise.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @return __m256 vector where each element is Xx[i] * Xy[i].
     */
    [[nodiscard]]
    inline vec8f mul(const vec8f Xx, const vec8f Xy) {
        return _mm256_mul_ps(Xx, Xy);
    }
    /**
     * @brief Divides two __m256 vectors element-wise.
     * @param Xx Numerator vector.
     * @param Xy Denominator vector.
     * @return __m256 vector where each element is Xx[i] / Xy[i].
     */
    [[nodiscard]]
    inline vec8f div(const vec8f Xx, const vec8f Xy) {
        return _mm256_div_ps(Xx, Xy);
    }

    /**
     * @brief Computes the exponential (e^x) for each element of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where each element is exp(x[i]).
     */
    [[nodiscard]]
    inline vec8f exp(const vec8f x) {
        return _mm256_exp_ps(x);
    }
    /**
     * @brief Computes the natural logarithm (ln(x)) for each element of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where each element is log(x[i]).
     */
    [[nodiscard]]
    inline vec8f log(const vec8f x) {
        return _mm256_log_ps(x);
    }
    /**
     * @brief Computes the square root for each element of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where each element is sqrt(x[i]).
     */
    [[nodiscard]]
    inline vec8f sqrt(const vec8f x) {
        return _mm256_sqrt_ps(x);
    }
    /**
     * @brief Computes the reciprocal square root (1 / sqrt(x)) for each element of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where each element is rsqrt(x[i]).
     * @note This is an approximate operation with lower precision but higher performance.
     */
    [[nodiscard]]
    inline vec8f rsqrt(const vec8f x) {
        return _mm256_rsqrt_ps(x);
    }
    /**
     * @brief Computes the power function (x^y) for each element of two __m256 vectors.
     * @param Xx Base vector.
     * @param Xy Exponent vector.
     * @return __m256 vector where each element is pow(Xx[i], Xy[i]).
     */
    [[nodiscard]]
    inline vec8f pow(const vec8f Xx, const vec8f Xy) {
        return _mm256_pow_ps(Xx, Xy);
    }
    /**
     * @brief Performs fused multiply-add operation on three __m256 vectors.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @param Xz Third input vector.
     * @return __m256 vector where each element is (Xx[i] * Xy[i]) + Xz[i].
     * @note Computed with a single rounding step for improved precision.
     */
    [[nodiscard]]
    inline vec8f fmadd(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
        return _mm256_fmadd_ps(Xx, Xy, Xz);
    }
    /**
     * @brief Performs fused multiply-subtract operation on three __m256 vectors.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @param Xz Third input vector.
     * @return __m256 vector where each element is (Xx[i] * Xy[i]) - Xz[i].
     * @note Computed with a single rounding step for improved precision.
     */
    [[nodiscard]]
    inline vec8f fmsub(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
        return _mm256_fmsub_ps(Xx, Xy, Xz);
    }
    /**
     * @brief Performs fused negative multiply-add operation on three __m256 vectors.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @param Xz Third input vector.
     * @return __m256 vector where each element is -(Xx[i] * Xy[i]) + Xz[i].
     * @note Computed with a single rounding step for improved precision.
     */
    [[nodiscard]]
    inline vec8f fnmadd(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
        return _mm256_fnmadd_ps(Xx, Xy, Xz);
    }
    /**
     * @brief Computes the absolute value for each element of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where each element is abs(x[i]).
     * @note Implemented using bitwise operations by clearing the sign bit.
     */
    [[nodiscard]]
    inline vec8f abs(const vec8f x) {
        return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    }
    /**
     * @brief Computes the negation for each element of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where each element is -x[i].
     * @note Implemented using bitwise XOR with the sign bit.
     */
    [[nodiscard]]
    inline vec8f neg(const vec8f x) {
        return _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    }
    /**
     * @brief Computes an approximate reciprocal (1 / x) using Newton-Raphson refinement.
     * @param x Input vector.
     * @return __m256 vector where each element is approximately 1 / x[i].
     * @note Uses one Newton-Raphson iteration to improve precision over _mm256_rcp_ps.
     */
    [[nodiscard]]
    inline vec8f rcp_nr(const vec8f x) {
        const vec8f r0 = _mm256_rcp_ps(x);
        return mul(r0, fnmadd(x, r0, set1(2.0f)));
    }
    template<i32 imm>
    [[nodiscard]]
    vec8f shuffle(const vec8f Xx, const vec8f Xy) {
        return _mm256_shuffle_ps(Xx, Xy, imm);
    }

    /**
     * @brief Selects elements from two __m256 vectors based on a mask.
     * @param Xx First input vector (selected when mask bit is 0).
     * @param Xy Second input vector (selected when mask bit is 1).
     * @param Xz Mask vector controlling the selection.
     * @return __m256 vector where each element is either Xx[i] or Xy[i].
     * @note Selection is based on the sign bit of each element in the mask.
     */
    [[nodiscard]]
    inline vec8f blendv(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
        return _mm256_blendv_ps(Xx, Xy, Xz);
    }
    /**
     * @brief Computes the element-wise maximum of two __m256 vectors.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @return __m256 vector where each element is max(Xx[i], Xy[i]).
     */
    [[nodiscard]]
    inline vec8f max(const vec8f Xx, const vec8f Xy) {
        return _mm256_max_ps(Xx, Xy);
    }
    /**
     * @brief Computes the element-wise minimum of two __m256 vectors.
     * @param Xx First input vector.
     * @param Xy Second input vector.
     * @return __m256 vector where each element is min(Xx[i], Xy[i]).
     */
    [[nodiscard]]
    inline vec8f min(const vec8f Xx, const vec8f Xy) {
        return _mm256_min_ps(Xx, Xy);
    }

    /**
     * @brief Applies the Rectified Linear Unit (ReLU) activation function.
     * @param x Input vector.
     * @return __m256 vector where each element is max(0, x[i]).
     */
    [[nodiscard]]
    inline vec8f relu(const vec8f x) {
        return max(zero(), x);
    }
    /**
     * @brief Computes the hyperbolic tangent for each element of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where each element is tanh(x[i]).
     */
    [[nodiscard]]
    inline vec8f tanh(const vec8f x) {
        return _mm256_tanh_ps(x);
    }
    /**
     * @brief Applies the Leaky ReLU activation function.
     * @param x Input vector.
     * @param alpha Slope for negative input values.
     * @return __m256 vector where each element is x[i] if x[i] > 0, otherwise alpha * x[i].
     */
    [[nodiscard]]
    inline vec8f leaky_relu(const vec8f x, const f32 alpha = 0.01f) {
        const vec8f mask = cmp::gt(x, zero());
        const vec8f neg_branch = mul(set1(alpha), x);
        return blendv(neg_branch, x, mask);
    }
    /**
     * @brief Computes the sigmoid activation function.
     * @param x Input vector.
     * @return __m256 vector where each element is 1 / (1 + exp(-x[i])).
     */
    [[nodiscard]]
    inline vec8f sigmoid(const vec8f x) {
        const vec8f neg_x = neg(x);
        const vec8f exp_neg = exp(neg_x);
        const vec8f denom = add(set1(1.0f), exp_neg);
        return div(set1(1.0f), denom);
    }
    /**
     * @brief Computes an optimized sigmoid using reciprocal approximation.
     * @param x Input vector.
     * @return __m256 vector where each element is approximately 1 / (1 + exp(-x[i])).
     * @note Faster than sigmoid() but with slightly reduced precision.
     */
    [[nodiscard]]
    inline vec8f sigmoid_fast(const vec8f x) {
        const vec8f exp_neg = exp(neg(x));
        return rcp_nr(add(set1(1.0f), exp_neg));
    }
    /**
     * @brief Computes the exact Gaussian Error Linear Unit (GELU) activation.
     * @param x Input vector.
     * @return __m256 vector where each element follows the exact GELU formulation.
     * @note Uses the error function (erf), which is more precise but slower.
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
     * @brief Computes an approximate Gaussian Error Linear Unit (GELU) activation.
     * @param x Input vector.
     * @return __m256 vector where each element follows an approximation of GELU.
     * @note Uses a tanh-based approximation for improved performance.
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
     * @brief Computes the Sigmoid Linear Unit (SiLU) activation function.
     * @param x Input vector.
     * @return __m256 vector where each element is x[i] * sigmoid(x[i]).
     */
    [[nodiscard]]
    inline vec8f silu(const vec8f x) {
        return mul(x, sigmoid(x));
    }
    /**
     * @brief Computes the Swish activation function.
     * @param x Input vector.
     * @param beta Scaling factor applied before the sigmoid.
     * @return __m256 vector where each element is x[i] * sigmoid(beta * x[i]).
     */
    [[nodiscard]]
    inline vec8f swish(const vec8f x, const f32 beta = 1.0f) {
        return mul(x, sigmoid(mul(set1(beta), x)));
    }
    /**
     * @brief Computes the softmax function over all elements of a __m256 vector.
     * @param x Input vector.
     * @return __m256 vector where elements represent normalized probabilities.
     * @note Applies max-shift for numerical stability.
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
     * @brief Computes the fast Sigmoid Linear Unit (SiLU) activation function.
     * @param x Input vector.
     * @return __m256 vector where each element is x[i] * sigmoid_fast(x[i]).
     * @note Faster than silu() but with slightly reduced precision.
     */
    [[nodiscard]]
    inline vec8f silu_fast(const vec8f x) {
        return mul(x, sigmoid_fast(x));
    }
    /**
     * @brief Computes the fast Swish activation function.
     * @param x Input vector.
     * @param beta Scaling factor applied before the sigmoid.
     * @return __m256 vector where each element is x[i] * sigmoid_fast(beta * x[i]).
     * @note Faster than swish() but with slightly reduced precision.
     */
    [[nodiscard]]
    inline vec8f swish_fast(const vec8f x, const f32 beta = 1.0f) {
        return mul(x, sigmoid_fast(mul(set1(beta), x)));
    }
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_FUNCTIONS_HPP