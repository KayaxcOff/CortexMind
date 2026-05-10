//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA accelerated matrix and vector operations.
     *
     * This struct contains both element-wise operations (using custom kernels)
     * and matrix multiplication (using cuBLAS).
     */
    struct Matrix {
        /**
         * @brief Element-wise addition: `Z[i] = X[i] + Y[i]`
         */
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise subtraction: `Z[i] = X[i] - Y[i]`
         */
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise multiplication: `Z[i] = X[i] * Y[i]`
         */
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise division: `Z[i] = X[i] / Y[i]`
         */
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);

        /**
         * @brief Matrix multiplication using cuBLAS: `Z = X × Y`
         *
         * Computes `Z[i,j] = sum_k X[i,k] * Y[k,j]` (row-major order).
         *
         * @param Xx Pointer to matrix A (shape: xN × yN)
         * @param Xy Pointer to matrix B (shape: yN × zN)
         * @param Xz Pointer to output matrix C (shape: xN × zN)
         * @param xN Rows of A and C
         * @param yN Columns of A = Rows of B
         * @param zN Columns of B and C
         *
         * @note Uses `cublasSgemm` with `CUBLAS_OP_N` (no transpose).
         */
        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);

        /**
         * @brief In-place addition: `X[i] += Y[i]`
         */
        static void add(f32* Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief In-place subtraction: `X[i] -= Y[i]`
         */
        static void sub(f32* Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief In-place multiplication: `X[i] *= Y[i]`
         */
        static void mul(f32* Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief In-place division: `X[i] /= Y[i]`
         */
        static void div(f32* Xx, const f32* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH