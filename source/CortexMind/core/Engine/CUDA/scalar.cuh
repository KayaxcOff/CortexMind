//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_CUH

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief   Launches vectorized scalar broadcast kernels on GPU
     *
     * Each method processes arrays in chunks of 4 floats using float4,
     * with scalar cleanup for the tail (idx % 4 elements).
     */
    struct ScalarKernel {
        /**
         * @brief   Z[i] = X[i] + value    (GPU, vectorized)
         * @param   Xx      Input array (device pointer)
         * @param   value   Scalar to add
         * @param   Xz      Output array (device pointer)
         * @param   idx     Number of float elements
         */
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] - value    (GPU, vectorized)
         * @param   Xx      Input array (device pointer)
         * @param   value   Scalar to add
         * @param   Xz      Output array (device pointer)
         * @param   idx     Number of float elements
         */
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] * value    (GPU, vectorized)
         * @param   Xx      Input array (device pointer)
         * @param   value   Scalar to add
         * @param   Xz      Output array (device pointer)
         * @param   idx     Number of float elements
         */
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] / value    (GPU, vectorized)
         * @warning value == 0 → Inf or NaN
         * @param   Xx      Input array (device pointer)
         * @param   value   Scalar to add
         * @param   Xz      Output array (device pointer)
         * @param   idx     Number of float elements
         */
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
    };
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_CUH