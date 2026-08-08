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
         * @param Xx Source array.
         * @param value Scalar operand.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void add(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Subs a scalar value to every element of the input array.
         *
         * @param Xx Source array.
         * @param value Scalar operand.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void sub(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Muls a scalar value to every element of the input array.
         *
         * @param Xx Source array.
         * @param value Scalar operand.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void mul(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Divs a scalar value to every element of the input array.
         *
         * @param Xx Source array.
         * @param value Scalar operand.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void div(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);

        /**
         * @brief Adds a scalar value to every element of the array in-place.
         *
         * @param Xx Input and output array.
         * @param value Scalar operand.
         * @param N Number of elements.
         */
        static void add(float* Xx, float value, std::size_t N);
        /**
         * @brief Subs a scalar value to every element of the array in-place.
         *
         * @param Xx Input and output array.
         * @param value Scalar operand.
         * @param N Number of elements.
         */
        static void sub(float* Xx, float value, std::size_t N);
        /**
         * @brief Muls a scalar value to every element of the array in-place.
         *
         * @param Xx Input and output array.
         * @param value Scalar operand.
         * @param N Number of elements.
         */
        static void mul(float* Xx, float value, std::size_t N);
        /**
         * @brief Divs a scalar value to every element of the array in-place.
         *
         * @param Xx Input and output array.
         * @param value Scalar operand.
         * @param N Number of elements.
         */
        static void div(float* Xx, float value, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP