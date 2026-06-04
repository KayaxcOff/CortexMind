//
// Created by muham on 8.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_ACTIVATION_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_ACTIVATION_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized activation functions for neural networks.
     *
     * All functions use a hybrid approach:
     * - Main loop: AVX2 vectorized path (using functions from `avx2::`)
     * - Remainder: Scalar fallback
     */
    struct Activation {
        /**
         * @brief ReLU activation: `Z[i] = max(0, X[i])`
         */
        static void relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Leaky ReLU activation: `Z[i] = X[i] > 0 ? X[i] : alpha * X[i]`
         *
         * @param alpha Negative slope coefficient (default = 0.01)
         */
        static void leaky_relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 alpha = 0.01f);
        /**
         * @brief Sigmoid activation: `Z[i] = 1 / (1 + e^(-X[i]))`
         */
        static void sigmoid(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Hyperbolic tangent activation: `Z[i] = tanh(X[i])`
         */
        static void tanh(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief GELU activation (approximate version using tanh).
         */
        static void gelu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief GELU activation using erf function (more accurate).
         */
        static void gelu_exact(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief SiLU (Swish with beta=1) activation: `Z[i] = X[i] * sigmoid(X[i])`
         */
        static void silu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Fast SiLU approximation.
         */
        static void silu_fast(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Swish activation: `Z[i] = X[i] * sigmoid(beta * X[i])`
         *
         * @param beta Scaling parameter (default = 1.0)
         */
        static void swish(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 beta = 1.0f);
        /**
         * @brief Fast Swish approximation.
         *
         * @param beta Scaling parameter (default = 1.0)
         */
        static void swish_fast(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 beta = 1.0f);
        /**
         * @brief Softmax activation along the vector.
         *
         * @note Computes `Z[i] = exp(X[i] - max) / sum(exp(X - max))`
         */
        static void softmax(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Fast Sigmoid approximation.
         */
        static void sigmoid_fast(const f32* Xx, f32* Xz, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_ACTIVATION_HPP