//
// Created by muham on 2.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP

#include <CortexMind/core/Engine/AVX/funcs.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2-accelerated matrix and vector arithmetic operations.
     *
     * This structure provides static helper functions for performing
     * element-wise and matrix-level arithmetic using AVX2 instructions.
     *
     * All index parameters use the project-specific `size` type, which is
     * conceptually equivalent to `std::size_t` and is intended to represent
     * logical element indices rather than raw byte offsets.
     *
     */
    struct matrix_t {
        /**
         * @brief Performs element-wise addition of two input arrays.
         *
         * Computes `_z = _x + _y` using AVX2 vectorized operations.
         *
         * @param _x  Pointer to the first input array.
         * @param _y  Pointer to the second input array.
         * @param _z  Pointer to the output array.
         * @param idx Logical element index indicating the starting position.
         *
         * @note Memory regions must be valid and large enough to accommodate
         *       the vectorized operation.
         */
        static void add(const f32* _x, const f32* _y, f32* _z, size_t idx);

        /**
        * @brief Performs element-wise subtraction of two input arrays.
        *
        * Computes `_z = _x - _y` using AVX2 vectorized operations.
        *
        * @param _x  Pointer to the first input array.
        * @param _y  Pointer to the second input array.
        * @param _z  Pointer to the output array.
        * @param idx Logical element index indicating the starting position.
        */
        static void sub(const f32* _x, const f32* _y, f32* _z, size_t idx);

        /**
         * @brief Performs element-wise multiplication of two input arrays.
         *
         * Computes `_z = _x * _y` using AVX2 vectorized operations.
         *
         * @param _x  Pointer to the first input array.
         * @param _y  Pointer to the second input array.
         * @param _z  Pointer to the output array.
         * @param idx Logical element index indicating the starting position.
         */
        static void mul(const f32* _x, const f32* _y, f32* _z, size_t idx);

        /**
         * @brief Performs element-wise division of two input arrays.
         *
         * Computes `_z = _x / _y` using AVX2 vectorized operations.
         *
         * @param _x  Pointer to the numerator array.
         * @param _y  Pointer to the denominator array.
         * @param _z  Pointer to the output array.
         * @param idx Logical element index indicating the starting position.
         *
         * @warning Division by zero is not checked and results in
         *          architecture-defined behavior.
         */
        static void div(const f32* _x, const f32* _y, f32* _z, size_t idx);

        /**
         * @brief Performs a fused multiply-add (FMA) operation.
         *
         * Computes `_m = (_x * _y) + _z` using AVX2 FMA instructions.
         *
         * @param _x  Pointer to the first multiplicand array.
         * @param _y  Pointer to the second multiplicand array.
         * @param _z  Pointer to the addend array.
         * @param _m  Pointer to the output array.
         * @param idx Logical element index indicating the starting position.
         */
        static void fma(const f32* _x, const f32* _y, const f32* _z, f32* _m, size_t idx);

        /**
         * @brief Performs matrix multiplication using AVX2 acceleration.
         *
         * Multiplies matrix `_x` with matrix `_y` and stores the result in `_z`.
         * Each index parameter specifies a logical element offset within its
         * respective matrix.
         *
         * @param _x   Pointer to the left-hand matrix.
         * @param _y   Pointer to the right-hand matrix.
         * @param _z   Pointer to the output matrix.
         * @param xIdx Logical element index for matrix `_x`.
         * @param yIdx Logical element index for matrix `_y`.
         * @param zIdx Logical element index for matrix `_z`.
         *
         */
        static void matmul(const f32* _x, const f32* _y, f32* _z, size_t xIdx, size_t yIdx, size_t zIdx);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP