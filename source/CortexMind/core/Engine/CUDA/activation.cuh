//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_CUH

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief   Launches in-place activation kernels on GPU
     *
     * All methods apply the corresponding activation function element-wise
     * and modify the input array directly (in-place).
     *
     * Typical usage:
     * @code
     *     cuda::activation_t::relu(d_data, n_elements);
     *     cudaDeviceSynchronize();  // if immediate result needed
     * @endcode
     */
    struct activation_t {
        /**
         * @brief   Applies ReLU in-place: X[i] = max(X[i], 0)
         * @param   Xx      Input/output array (device pointer)
         * @param   idx     Number of elements
         */
        static void relu(f32* __restrict Xx, size_t idx);
        /**
         * @brief   Applies Leaky ReLU in-place: X[i] = X[i] > 0 ? X[i] : alpha * X[i]
         * @param   Xx      Input/output array
         * @param   alpha   Negative slope (typical: 0.01 or 0.1)
         * @param   idx     Number of elements
         */
        static void leaky_relu(f32* __restrict Xx, f32 alpha, size_t idx);
        /**
         * @brief   Applies Sigmoid in-place: X[i] = 1 / (1 + exp(-X[i]))
         * @param   Xx      Input/output array
         * @param   idx     Number of elements
         */
        static void sigmoid(f32* __restrict Xx, size_t idx);
        /**
         * @brief   Applies fast Sigmoid approximation in-place
         * @param   Xx      Input/output array
         * @param   idx     Number of elements
         */
        static void sigmoid_fast(f32* __restrict Xx, size_t idx);
        /**
         * @brief   Applies approximate GELU in-place
         * @param   Xx      Input/output array
         * @param   idx     Number of elements
         */
        static void gelu(f32* __restrict Xx, size_t idx);
        /**
         * @brief   Applies exact GELU (using erf) in-place
         * @param   Xx      Input/output array
         * @param   idx     Number of elements
         * @note    Requires erff support (may need -use_fast_math or SVML)
         */
        static void gelu_exact(f32* __restrict Xx, size_t idx);
        /**
         * @brief   Applies SiLU (Sigmoid Linear Unit) in-place: X[i] × sigmoid(X[i])
         * @param   Xx      Input/output array
         * @param   idx     Number of elements
         */
        static void silu(f32* __restrict Xx, size_t idx);
        /**
         * @brief   Applies Swish (parameterized): X[i] × sigmoid(β × X[i])
         * @param   Xx      Input/output array
         * @param   beta    Scaling factor (typical: 1.0)
         * @param   idx     Number of elements
         */
        static void swish(f32* __restrict Xx, f32 beta, size_t idx);
        static void exp(f32* __restrict__ Xx, size_t idx);
        static void log(f32* __restrict__ Xx, size_t idx);
        static void abs(f32* __restrict__ Xx, size_t idx);
    };
} // namespace cortex::_fw::cuda

#endif // CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_CUH