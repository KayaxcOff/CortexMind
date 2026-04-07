//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_ACTIVATION_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_ACTIVATION_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized activation functions for neural networks.
     *
     * Provides both out-of-place and in-place versions of common activation functions.
     * Vectorized AVX2 implementations are used for full 8-element chunks with scalar fallback.
     */
    struct Activation {
        /**
         * @brief Applies the ReLU (Rectified Linear Unit) activation function.
         *
         * Computes element-wise: max(0, x)
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         */
        static void relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Applies the Leaky ReLU activation function.
         *
         * Computes element-wise:
         *   x       if x > 0
         *   alpha*x otherwise
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         * @param alpha Slope for negative values.
         */
        static void leaky_relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 alpha = 0.01f);
        /**
         * @brief Applies the Sigmoid activation function.
         *
         * Computes element-wise: 1 / (1 + exp(-x))
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         */
        static void sigmoid(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Applies the hyperbolic tangent (tanh) activation function.
         *
         * Computes element-wise: tanh(x)
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         */
        static void tanh(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Applies the approximate GELU (Gaussian Error Linear Unit).
         *
         * Uses a fast tanh-based approximation.
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         *
         * @note Faster than gelu_exact(), but slightly less numerically precise.
         */
        static void gelu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Applies the exact GELU activation function.
         *
         * Uses the error function (erf) for higher numerical accuracy.
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         *
         * @note More precise than gelu(), but computationally slower.
         */
        static void gelu_exact(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Applies the SiLU (Sigmoid Linear Unit), also known as Swish-1.
         *
         * Computes element-wise: x * sigmoid(x)
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         */
        static void silu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Applies a fast approximation of the SiLU activation.
         *
         * Uses a faster internal approximation for improved performance.
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         *
         * @note Faster than silu(), but may introduce small numerical deviations.
         */
        static void silu_fast(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Applies the Swish activation function.
         *
         * Computes element-wise: x * sigmoid(beta * x)
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         * @param beta Scaling factor.
         */
        static void swish(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 beta = 1.0f);
        /**
         * @brief Applies a fast approximation of the Swish activation.
         *
         * Uses a faster approximation for sigmoid computation.
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         * @param beta Scaling factor.
         *
         * @note Faster than swish(), but may slightly reduce numerical accuracy.
         */
        static void swish_fast(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 beta = 1.0f);
        /**
         * @brief Applies the Softmax activation function.
         *
         * Computes a normalized exponential over the input array:
         *   softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
         *
         * A max-subtraction trick is used for numerical stability.
         *
         * @param Xx Input array pointer.
         * @param Xz Output array pointer.
         * @param N  Number of elements.
         */
        static void softmax(const f32* __restrict Xx, f32* __restrict Xz, size_t N);

        /**
         * @brief In-place ReLU activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         */
        static void relu(f32* Xx, size_t N);
        /**
         * @brief In-place Leaky ReLU activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         * @param alpha Slope for negative values.
         */
        static void leaky_relu(f32* Xx, size_t N, f32 alpha = 0.01f);
        /**
         * @brief In-place Sigmoid activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         */
        static void sigmoid(f32* Xx, size_t N);
        /**
         * @brief In-place tanh activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         */
        static void tanh(f32* Xx, size_t N);
        /**
         * @brief In-place approximate GELU activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         *
         * @note Faster but less precise than gelu_exact().
         */
        static void gelu(f32* Xx, size_t N);
        /**
         * @brief In-place exact GELU activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         *
         * @note More accurate but slower than gelu().
         */
        static void gelu_exact(f32* Xx, size_t N);
        /**
         * @brief In-place SiLU activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         */
        static void silu(f32* Xx, size_t N);
        /**
         * @brief In-place fast SiLU activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         *
         * @note Faster than silu(), with minor precision trade-offs.
         */
        static void silu_fast(f32* Xx, size_t N);
        /**
         * @brief In-place Swish activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         * @param beta Scaling factor.
         */
        static void swish(f32* Xx, size_t N, f32 beta = 1.0f);
        /**
         * @brief In-place fast Swish activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         * @param beta Scaling factor.
         *
         * @note Faster than swish(), but slightly less precise.
         */
        static void swish_fast(f32* Xx, size_t N, f32 beta = 1.0f);
        /**
         * @brief In-place Softmax activation.
         *
         * @param Xx Input/output array.
         * @param N  Number of elements.
         */
        static void softmax(f32* Xx, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_ACTIVATION_HPP