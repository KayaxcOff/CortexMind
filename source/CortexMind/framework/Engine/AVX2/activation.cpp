//
// Created by muham on 8.05.2026.
//

#include "CortexMind/framework/Engine/AVX2/activation.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/AVX2/horizontal.hpp>
#include <CortexMind/framework/Engine/AVX2/reduce.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

namespace {
    /**
     * @brief Computes the sigmoid activation function for a scalar value.
     *
     * The sigmoid function maps the input value into the range (0, 1)
     * using the following formula:
     *
     * \f[
     * \sigma(x) = \frac{1}{1 + e^{-x}}
     * \f]
     *
     * Commonly used in logistic regression and neural network activations.
     *
     * @param x Input scalar value.
     * @return Sigmoid-transformed value.
     */
    f32 scalar_sigmoid(const f32 x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    /**
     * @brief Computes the approximate GELU (Gaussian Error Linear Unit) activation.
     *
     * This implementation uses the tanh-based approximation introduced in
     * the original GELU paper:
     *
     * \f[
     * \text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(
     * \sqrt{\frac{2}{\pi}} (x + 0.044715x^3)
     * \right)\right)
     * \f]
     *
     * The approximation is faster than the exact formulation and is widely
     * used in transformer-based neural networks.
     *
     * @param x Input scalar value.
     * @return Approximate GELU activation output.
     */
    f32 scalar_gelu(const f32 x) {
        constexpr f32 c0 = 0.7978845608028654f;
        constexpr f32 c1 = 0.044715f;
        return 0.5f * x * (1.0f + std::tanh(c0 * (x + c1 * x * x * x)));
    }
    /**
     * @brief Computes the exact GELU (Gaussian Error Linear Unit) activation.
     *
     * This implementation uses the error function (`std::erf`) to evaluate
     * the exact GELU formulation:
     *
     * \f[
     * \text{GELU}(x) =
     * 0.5x \left(1 + \operatorname{erf}\left(
     * \frac{x}{\sqrt{2}}
     * \right)\right)
     * \f]
     *
     * Compared to the approximate version, this implementation is more accurate
     * but may be slower depending on the platform and standard library.
     *
     * @param x Input scalar value.
     * @return Exact GELU activation output.
     */
    f32 scalar_gelu_exact(const f32 x) {
        return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
    }
    /**
     * @brief Computes a fast sigmoid approximation for a scalar value.
     *
     * This variant caches the exponential computation in a temporary variable
     * before evaluating the sigmoid expression:
     *
     * \f[
     * \sigma(x) = \frac{1}{1 + e^{-x}}
     * \f]
     *
     * Functionally equivalent to `scalar_sigmoid`.
     *
     * @param x Input scalar value.
     * @return Sigmoid-transformed value.
     */
    f32 scalar_sigmoid_fast(const f32 x) {
        const f32 e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
} //unnamed namespace

void Activation::relu(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::relu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::max(0.0f, Xx[i]);
    }
}

void Activation::leaky_relu(const f32* Xx, f32* Xz, const size_t N, const f32 alpha) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::leaky_relu(loadu(Xx + i), alpha));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] > 0.0f ? Xx[i] : alpha * Xx[i];
    }
}

void Activation::sigmoid(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sigmoid(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = scalar_sigmoid(Xx[i]);
    }
}

void Activation::tanh(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::tanh(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::tanh(Xx[i]);
    }
}

void Activation::gelu(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::gelu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = scalar_gelu(Xx[i]);
    }
}

void Activation::gelu_exact(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::gelu_exact(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = scalar_gelu_exact(Xx[i]);
    }
}

void Activation::silu(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::silu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(Xx[i]);
    }
}

void Activation::silu_fast(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::silu_fast(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(Xx[i]);
    }
}

void Activation::swish(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::swish(loadu(Xx + i), beta));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(beta * Xx[i]);
    }
}

void Activation::swish_fast(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::swish_fast(loadu(Xx + i), beta));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(beta * Xx[i]);
    }
}

void Activation::softmax(const f32* Xx, f32* Xz, const size_t N) {
    f32 max_val = 0.0f;
    reduce::max(Xx, &max_val, N);
    const vec8f vmax = set1(max_val);

    vec8f vacc = zero();
    size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        const vec8f shifted = sub(loadu(Xx + i), vmax);
        const vec8f exp_vals = avx2::exp(shifted);
        storeu(Xz + i, exp_vals);
        vacc = add(vacc, exp_vals);
    }

    f32 tail_sum = 0.0f;
    for (; i < N; ++i) {
        const f32 exp_val = std::exp(Xx[i] - max_val);
        Xz[i] = exp_val;
        tail_sum += exp_val;
    }

    f32 total_sum = horizontal::sum(vacc) + tail_sum;
    total_sum = std::max(total_sum, 1e-10f);

    const f32 inv_sum = 1.0f / total_sum;
    const vec8f v_inv_sum = set1(inv_sum);

    i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, mul(loadu(Xz + i), v_inv_sum));
    }
    for (; i < N; ++i) {
        Xz[i] *= inv_sum;
    }
}

void Activation::sigmoid_fast(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sigmoid_fast(loadu(Xx + i)));
    }

    for (; i < N; ++i) {
        Xz[i] = scalar_sigmoid_fast(Xx[i]);
    }
}