//
// Created by muham on 8.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized matrix and vector operations.
     *
     * This struct contains both element-wise operations and a high-performance
     * matrix multiplication (matmul) implementation.
     */
    struct matrix_t {
        /**
         * @brief Out-of-place element-wise addition: `Z[i] = X[i] + Y[i]`
         *
         * @param Xx First input array
         * @param Xy Second input array
         * @param Xz Output array
         * @param N  Number of elements
         */
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place element-wise subtraction: `Z[i] = X[i] - Y[i]`
         *
         * @param Xx First input array
         * @param Xy Second input array
         * @param Xz Output array
         * @param N  Number of elements
         */
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place element-wise multiplication: `Z[i] = X[i] * Y[i]`
         *
         * @param Xx First input array
         * @param Xy Second input array
         * @param Xz Output array
         * @param N  Number of elements
         */
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place element-wise division: `Z[i] = X[i] / Y[i]`
         *
         * @param Xx First input array
         * @param Xy Second input array
         * @param Xz Output array
         * @param N  Number of elements
         */
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);

        /**
         * @brief Matrix multiplication (C = A × B) with blocked cache-friendly algorithm.
         *
         * @param Xx Pointer to matrix A (shape: xN × yN)
         * @param Xy Pointer to matrix B (shape: yN × zN)
         * @param Xz Pointer to output matrix C (shape: xN × zN)
         * @param xN Number of rows in A and C
         * @param yN Number of columns in A = rows in B
         * @param zN Number of columns in B and C
         *
         * @note Uses a 3-level blocked (tiled) matrix multiplication algorithm
         *       with micro-kernel for better cache utilization and register blocking.
         * @warning Output matrix `Xz` is zero-initialized inside the function.
         */
        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);

        /**
         * @brief In-place element-wise addition: `X[i] = X[i] + Y[i]`
         *
         * @param Xx Input and output array
         * @param Xy Input array
         * @param N  Number of elements
         */
        static void add(f32* Xx, const f32* __restrict Xy, size_t N);

        /**
         * @brief In-place element-wise subtraction: `X[i] = X[i] - Y[i]`
         *
         * @param Xx Input and output array
         * @param Xy Input array
         * @param N  Number of elements
         */
        static void sub(f32* Xx, const f32* __restrict Xy, size_t N);

        /**
         * @brief In-place element-wise multiplication: `X[i] = X[i] * Y[i]`
         *
         * @param Xx Input and output array
         * @param Xy Input array
         * @param N  Number of elements
         */
        static void mul(f32* Xx, const f32* __restrict Xy, size_t N);

        /**
         * @brief In-place element-wise division: `X[i] = X[i] / Y[i]`
         *
         * @param Xx Input and output array
         * @param Xy Input array
         * @param N  Number of elements
         */
        static void div(f32* Xx, const f32* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::avx2 {}

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP