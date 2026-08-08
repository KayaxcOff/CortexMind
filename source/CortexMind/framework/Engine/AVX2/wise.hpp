//
// Created by muham on 7.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP

#include <cwchar>

namespace cortex::_fw::avx2 {
    /**
     * @brief Element-wise mathematical operations for contiguous float arrays.
     *
     * Provides vectorized mathematical transformations over contiguous
     * single-precision floating-point arrays.
     *
     * Operations are implemented using the CortexMind AVX2 primitives for
     * the vectorized portion of the input and scalar operations for any
     * remaining tail elements.
     *
     * Unless otherwise specified, operations write their results to the
     * destination buffer without modifying the source buffer.
     */
    struct wise {
        /**
         * @brief Computes the element-wise square of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void square(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Raises each element of an array to a scalar power.
         *
         * @param Xx Source array.
         * @param value Scalar exponent.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void pow(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        /**
         * @brief Raises each element of one array to the corresponding element of another array.
         *
         * Computes `Xz[i] = pow(Xx[i], Xy[i])` for each element.
         *
         * @param Xx Base array.
         * @param Xy Exponent array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void pow(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise square root of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void sqrt(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise reciprocal square root of an array.
         *
         * Computes the reciprocal of the square root for each element.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void rsqrt(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise natural logarithm of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void log(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise exponential of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void exp(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise error function of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void erf(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise sine of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void sin(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise cosine of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void cos(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise absolute value of an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void abs(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise negation of an array.
         *
         * Computes `Xz[i] = -Xx[i]` for each element.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void neg(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise reciprocal of an array.
         *
         * Computes `Xz[i] = 1 / Xx[i]` for each element.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void rcp(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the element-wise reciprocal of an array.
         *
         * Computes `Xz[i] = 1 / Xx[i]` for each element.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void inverse(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Performs element-wise linear interpolation.
         *
         * Interpolates each source element between @p value1 and @p value2.
         *
         * @param Xx Source array containing interpolation factors.
         * @param value1 Lower interpolation value.
         * @param value2 Upper interpolation value.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void lerp(const float* __restrict Xx, float value1, float value2, float* __restrict Xz, std::size_t N);
        /**
         * @brief Clamps each element of an array to a specified range.
         *
         * Values smaller than @p min are replaced by @p min, while values
         * greater than @p max are replaced by @p max.
         *
         * @param Xx Source array.
         * @param min Lower bound.
         * @param max Upper bound.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void clamp(const float* __restrict Xx, float min, float max, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the sign of each element in an array.
         *
         * @param Xx Source array.
         * @param Xz Destination array.
         * @param N Number of elements.
         */
        static void sign(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP