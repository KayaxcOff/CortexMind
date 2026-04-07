//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized matrix and vector arithmetic operations.
     *
     * Provides element-wise addition, subtraction, multiplication, division,
     * fused multiply-add/subtract, and matrix multiplication for float arrays.
     * Uses vectorized AVX2 instructions with scalar fallback for remaining elements.
     */
    struct matrix_t {
        /**
        * @brief Z = Xx + Xy (out-of-place)
        */
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
        * @brief Z = Xx - Xy (out-of-place)
        */
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
        * @brief Z = Xx * Xy (out-of-place)
        */
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        /**
        * @brief Z = Xx / Xy (out-of-place)
        */
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);

        /**
         * @brief Xk = Xx * Xy + Xz (out-of-place fused multiply-add)
         */
        static void fmadd(const f32* __restrict Xx, const f32* __restrict Xy, const f32* __restrict Xz, f32* __restrict Xk, size_t N);
        /**
         * @brief Xk = Xx * Xy - Xz (out-of-place fused multiply-subtract)
         */
        static void fmsub(const f32* __restrict Xx, const f32* __restrict Xy, const f32* __restrict Xz, f32* __restrict Xk, size_t N);

        /**
         * @brief General matrix multiplication: Z = X * Y (X: xN×yN, Y: yN×zN, Z: xN×zN)
         *
         * Uses tiled and vectorized implementation with AVX2 for high performance.
         */
        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);

        /**
         * @brief X = X + Y (in-place)
         */
        static void add(f32* Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief X = X - Y (in-place)
         */
        static void sub(f32* Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief X = X * Y (in-place)
         */
        static void mul(f32* Xx, const f32* __restrict Xy, size_t N);
        /**
         * @brief X = X / Y (in-place)
         */
        static void div(f32* Xx, const f32* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP