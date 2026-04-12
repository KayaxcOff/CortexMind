//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H
#define CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief High-level interface for CUDA scalar arithmetic kernels.
     *
     * Provides convenient static functions to launch optimized CUDA kernels
     * that perform element-wise operations between a vector and a scalar value
     * (addition, subtraction, multiplication, division).
     */
    struct ScalarKernel {
        /**
         * @brief Launches CUDA kernel for Z = X + value (out-of-place).
         */
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Launches CUDA kernel for Z = X - value (out-of-place).
         */
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Launches CUDA kernel for Z = X * value (out-of-place).
         */
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Launches CUDA kernel for Z = X / value (out-of-place).
         */
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);

        /**
         * @brief X = X + value (in-place)
         */
        static void add(f32* Xx, f32 value, size_t N);
        /**
         * @brief X = X - value (in-place)
         */
        static void sub(f32* Xx, f32 value, size_t N);
        /**
         * @brief X = X * value (in-place)
         */
        static void mul(f32* Xx, f32 value, size_t N);
        /**
         * @brief X = X / value (in-place)
         */
        static void div(f32* Xx, f32 value, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H