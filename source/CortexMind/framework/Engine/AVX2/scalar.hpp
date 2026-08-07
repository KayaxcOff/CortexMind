//
// Created by muham on 6.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP

#include <cstdio>

namespace cortex::_fw::avx2 {
    /**
     * @brief Performs scalar arithmetic on contiguous float arrays using AVX2.
     *
     * Provides element-wise arithmetic between an input array and a scalar
     * value. Operations are implemented using AVX2 vectorization whenever
     * possible and automatically fall back to scalar processing for the
     * remaining tail elements.
     *
     * Two operation modes are provided:
     * - Out-of-place operations producing a separate destination array.
     * - In-place operations modifying the source array directly.
     */
    struct ScalarOp {
        /**
         * @brief Adds a scalar value to every element of the input array.
         *
         * @param x1 Source array.
         * @param value Scalar operand.
         * @param x2 Destination array.
         * @param n Number of elements.
         */
        static void add(const float* __restrict x1, float value, float* __restrict x2, std::size_t n);
        /**
         * @brief Subs a scalar value to every element of the input array.
         *
         * @param x1 Source array.
         * @param value Scalar operand.
         * @param x2 Destination array.
         * @param n Number of elements.
         */
        static void sub(const float* __restrict x1, float value, float* __restrict x2, std::size_t n);
        /**
         * @brief Muls a scalar value to every element of the input array.
         *
         * @param x1 Source array.
         * @param value Scalar operand.
         * @param x2 Destination array.
         * @param n Number of elements.
         */
        static void mul(const float* __restrict x1, float value, float* __restrict x2, std::size_t n);
        /**
         * @brief Divs a scalar value to every element of the input array.
         *
         * @param x1 Source array.
         * @param value Scalar operand.
         * @param x2 Destination array.
         * @param n Number of elements.
         */
        static void div(const float* __restrict x1, float value, float* __restrict x2, std::size_t n);

        /**
         * @brief Adds a scalar value to every element of the array in-place.
         *
         * @param x0 Input and output array.
         * @param value Scalar operand.
         * @param n Number of elements.
         */
        static void add(float* x0, float value, std::size_t n);
        /**
         * @brief Subs a scalar value to every element of the array in-place.
         *
         * @param x0 Input and output array.
         * @param value Scalar operand.
         * @param n Number of elements.
         */
        static void sub(float* x0, float value, std::size_t n);
        /**
         * @brief Muls a scalar value to every element of the array in-place.
         *
         * @param x0 Input and output array.
         * @param value Scalar operand.
         * @param n Number of elements.
         */
        static void mul(float* x0, float value, std::size_t n);
        /**
         * @brief Divs a scalar value to every element of the array in-place.
         *
         * @param x0 Input and output array.
         * @param value Scalar operand.
         * @param n Number of elements.
         */
        static void div(float* x0, float value, std::size_t n);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP