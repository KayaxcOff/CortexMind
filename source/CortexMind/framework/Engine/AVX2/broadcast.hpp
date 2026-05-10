//
// Created by muham on 8.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized broadcast operations (row-wise and column-wise).
     *
     * Supports both out-of-place and in-place operations with hybrid
     * vectorized + scalar fallback implementation.
     */
    struct Broadcast {
        /**
         * @brief Out-of-place row-wise addition: `Z[row] = X[row] + Y`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length N)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void row_add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        /**
         * @brief Out-of-place row-wise subtraction: `Z[row] = X[row] - Y`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length N)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void row_sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        /**
         * @brief Out-of-place row-wise multiplication: `Z[row] = X[row] * Y`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length N)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void row_mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        /**
         * @brief Out-of-place row-wise division: `Z[row] = X[row] / Y`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length N)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void row_div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);

        /**
         * @brief In-place row-wise addition: `X[row] += Y`
         */
        static void row_add(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        /**
         * @brief In-place row-wise subtraction: `X[row] -= Y`
         */
        static void row_sub(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        /**
         * @brief In-place row-wise multiplication: `X[row] *= Y`
         */
        static void row_mul(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        /**
         * @brief In-place row-wise division: `X[row] /= Y`
         */
        static void row_div(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        /**
         * @brief Out-of-place column-wise addition: `Z[i,j] = X[i,j] + Y[i]`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length M)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void col_add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        /**
         * @brief Out-of-place column-wise subtraction: `Z[i,j] = X[i,j] - Y[i]`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length M)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void col_sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        /**
         * @brief Out-of-place column-wise multiplication: `Z[i,j] = X[i,j] * Y[i]`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length M)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void col_mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        /**
         * @brief Out-of-place column-wise division: `Z[i,j] = X[i,j] / Y[i]`
         *
         * @param Xx Input matrix A (M × N)
         * @param Xy Input vector (length M)
         * @param Xz Output matrix (M × N)
         * @param M  Number of rows
         * @param N  Number of columns
         */
        static void col_div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);

        /**
         * @brief In-place column-wise addition: `X[i,j] += Y[i]`
         */
        static void col_add(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        /**
         * @brief In-place column-wise subtraction: `X[i,j] -= Y[i]`
         */
        static void col_sub(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        /**
         * @brief In-place column-wise multiplication: `X[i,j] *= Y[i]`
         */
        static void col_mul(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        /**
         * @brief In-place column-wise division: `X[i,j] /= Y[i]`
         */
        static void col_div(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP