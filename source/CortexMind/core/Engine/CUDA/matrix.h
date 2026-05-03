//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H
#define CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief High-level interface for CUDA matrix and vector arithmetic operations.
     *
     * Provides optimized element-wise operations (add, sub, mul, div) and
     * matrix multiplication using cuBLAS (SGEMM).
     */
    struct Matrix {
        /**
         * @brief Z = Xx + Xy (out-of-place, element-wise)
         */
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Z = Xx - Xy (out-of-place, element-wise)
         */
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Z = Xx * Xy (out-of-place, element-wise)
         */
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Z = Xx / Xy (out-of-place, element-wise)
         */
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief General matrix multiplication: Z = Xx × Xy using cuBLAS.
         *
         * Matrix dimensions:
         * - Xx : xN × yN
         * - Xy : yN × zN
         * - Z  : xN × zN
         *
         * @note Uses cuBLAS SGEMM with column-major storage assumption.
         */
        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);

        /**
         * @brief X = X + Y (in-place)
         */
        static void add(f32* __restrict Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief X = X - Y (in-place)
         */
        static void sub(f32* __restrict Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief X = X * Y (in-place)
         */
        static void mul(f32* __restrict Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief X = X / Y (in-place)
         */
        static void div(f32* __restrict Xx, const f32* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H