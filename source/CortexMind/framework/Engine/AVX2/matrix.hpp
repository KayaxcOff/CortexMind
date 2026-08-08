//
// Created by muham on 8.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP

#include <cwchar>

namespace cortex::_fw::avx2 {
    /**
     * @brief Element-wise and matrix operations for contiguous float arrays.
     *
     * Provides vectorized binary arithmetic operations, element-wise
     * minimum and maximum operations, and matrix multiplication for
     * single-precision floating-point data.
     *
     * Element-wise operations support both out-of-place and in-place
     * execution. The vectorized portion is processed using AVX2
     * instructions, while remaining elements are handled by scalar
     * operations.
     *
     * Matrix multiplication uses a blocked AVX2 implementation with
     * fused multiply-add operations and masked processing for incomplete
     * tiles.
     */
    struct matrix_t {
        /**
         * @brief Computes element-wise addition of two arrays.
         *
         * Computes `Xz[i] = Xx[i] + Xy[i]` for each element.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes element-wise subtraction of two arrays.
         *
         * Computes `Xz[i] = Xx[i] - Xy[i]` for each element.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes element-wise multiplication of two arrays.
         *
         * Computes `Xz[i] = Xx[i] * Xy[i]` for each element.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes element-wise division of two arrays.
         *
         * Computes `Xz[i] = Xx[i] / Xy[i]` for each element.
         *
         * @param Xx Dividend array.
         * @param Xy Divisor array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);

        /**
         * @brief Computes the element-wise maximum of two arrays.
         *
         * Computes `Xz[i] = max(Xx[i], Xy[i])` for each element.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void max(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise minimum of two arrays.
         *
         * Computes `Xz[i] = min(Xx[i], Xy[i])` for each element.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void min(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);

        /**
         * @brief Computes the matrix product of two row-major matrices.
         *
         * Computes the matrix product
         *
         * `Xz = Xx * Xy`
         *
         * where `Xx` has dimensions `xN × yN`, `Xy` has dimensions
         * `yN × zN`, and `Xz` has dimensions `xN × zN`.
         *
         * The implementation uses cache-blocked matrix multiplication with
         * AVX2 vectorization and fused multiply-add operations. Incomplete
         * tiles at matrix boundaries are processed using runtime masks.
         *
         * All matrices are expected to use row-major contiguous storage.
         *
         * @param Xx Left-hand input matrix with dimensions `xN × yN`.
         * @param Xy Right-hand input matrix with dimensions `yN × zN`.
         * @param Xz Destination matrix with dimensions `xN × zN`.
         * @param xN Number of rows in the first matrix.
         * @param yN Shared dimension between the two matrices.
         * @param zN Number of columns in the second matrix.
         */
        static void matmul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t xN, std::size_t yN, std::size_t zN);

        /**
         * @brief Adds two arrays element-wise in-place.
         *
         * Computes `Xx[i] += Xy[i]` for each element.
         *
         * @param Xx Input and output array.
         * @param Xy Second input array.
         * @param N Number of elements.
         */
        static void add(float* Xx, const float* __restrict Xy, std::size_t N);
        /**
         * @brief Subtracts one array from another element-wise in-place.
         *
         * Computes `Xx[i] -= Xy[i]` for each element.
         *
         * @param Xx Input and output array.
         * @param Xy Array to subtract.
         * @param N Number of elements.
         */
        static void sub(float* Xx, const float* __restrict Xy, std::size_t N);
        /**
         * @brief Multiplies two arrays element-wise in-place.
         *
         * Computes `Xx[i] *= Xy[i]` for each element.
         *
         * @param Xx Input and output array.
         * @param Xy Second input array.
         * @param N Number of elements.
         */
        static void mul(float* Xx, const float* __restrict Xy, std::size_t N);
        /**
         * @brief Divides an array by another array element-wise in-place.
         *
         * Computes `Xx[i] /= Xy[i]` for each element.
         *
         * @param Xx Input and output array.
         * @param Xy Divisor array.
         * @param N Number of elements.
         */
        static void div(float* Xx, const float* __restrict Xy, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP