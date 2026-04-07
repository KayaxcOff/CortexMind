//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Scalar-vector operations with AVX2 optimization and scalar fallback.
     *
     * Provides optimized addition, subtraction, multiplication and division
     * between a vector and a scalar value. Uses AVX2 for full 8-element chunks
     * and falls back to scalar operations for the remaining elements.
     */
    struct ScalarOp {
        /**
         * @brief Z = X + value (out-of-place)
         */
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Z = X - value (out-of-place)
         */
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Z = X * value (out-of-place)
         */
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Z = X / value (out-of-place)
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
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP