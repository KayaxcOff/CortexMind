//
// Created by muham on 26.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STD_SCALAR_HPP
#define CORTEXMIND_CORE_ENGINE_STD_SCALAR_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::stl {
    /**
     * @brief Standard (scalar) scalar-vector operations.
     *
     * Provides simple, non-vectorized implementations for scalar operations.
     * Used as fallback or for platforms without AVX2 support.
     */
    struct Scalar {
        /**
         * @brief Out-of-place addition: Z = X + value
         * @param Xx Input array
         * @param value Scalar value
         * @param Xz Output array
         * @param N Number of elements
         */
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place subtraction: Z = X - value
         * @param Xx Input array
         * @param value Scalar value
         * @param Xz Output array
         * @param N Number of elements
         */
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place multiplication: Z = X * value
         * @param Xx Input array
         * @param value Scalar value
         * @param Xz Output array
         * @param N Number of elements
         */
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place division: Z = X / value
         * @param Xx Input array
         * @param value Scalar value
         * @param Xz Output array
         * @param N Number of elements
         */
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);

        /**
         * @brief In-place addition: X = X + value
         * @param Xx Input/Output array
         * @param value Scalar value
         * @param N Number of elements
         */
        static void add(f32* Xx, f32 value, size_t N);
        /**
         * @brief In-place subtraction: X = X - value
         * @param Xx Input/Output array
         * @param value Scalar value
         * @param N Number of elements
         */
        static void sub(f32* Xx, f32 value, size_t N);
        /**
         * @brief In-place multiplication: X = X * value
         * @param Xx Input/Output array
         * @param value Scalar value
         * @param N Number of elements
         */
        static void mul(f32* Xx, f32 value, size_t N);
        /**
         * @brief In-place division: X = X / value
         * @param Xx Input/Output array
         * @param value Scalar value
         * @param N Number of elements
         */
        static void div(f32* Xx, f32 value, size_t N);
    };
} //namespace cortex::_fw::stl

#endif //CORTEXMIND_CORE_ENGINE_STD_SCALAR_HPP