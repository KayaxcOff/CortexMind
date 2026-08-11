//
// Created by muham on 10.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_COMPARE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_COMPARE_HPP

#include <cstdint>
#include <cstddef>

namespace cortex::_fw::avx2 {
    /**
     * @brief Performs element-wise floating-point comparison operations.
     *
     * Provides AVX2-accelerated comparison operations between two contiguous
     * arrays of single-precision floating-point values. Each comparison
     * produces an `int32` result where `1` represents a true comparison and
     * `0` represents a false comparison.
     *
     * The comparison results are written to a separate output buffer.
     */
    struct compare {
        /**
         * @brief Tests whether each element of the first input is greater than the second.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Output array containing `1` for true comparisons and `0` otherwise.
         * @param N Number of elements to process.
         */
        static void gt(const float* __restrict Xx, const float* __restrict Xy, std::int32_t* __restrict Xz, std::size_t N);

        /**
         * @brief Tests whether each element of the first input is less than the second.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Output array containing `1` for true comparisons and `0` otherwise.
         * @param N Number of elements to process.
         */
        static void lt(const float* __restrict Xx, const float* __restrict Xy, std::int32_t* __restrict Xz, std::size_t N);

        /**
         * @brief Tests whether each pair of elements is equal.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Output array containing `1` for true comparisons and `0` otherwise.
         * @param N Number of elements to process.
         */
        static void eq(const float* __restrict Xx, const float* __restrict Xy, std::int32_t* __restrict Xz, std::size_t N);

        /**
         * @brief Tests whether each element of the first input is greater than or equal to the second.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Output array containing `1` for true comparisons and `0` otherwise.
         * @param N Number of elements to process.
         */
        static void ge(const float* __restrict Xx, const float* __restrict Xy, std::int32_t* __restrict Xz, std::size_t N);

        /**
         * @brief Tests whether each element of the first input is less than or equal to the second.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Output array containing `1` for true comparisons and `0` otherwise.
         * @param N Number of elements to process.
         */
        static void le(const float* __restrict Xx, const float* __restrict Xy, std::int32_t* __restrict Xz, std::size_t N);

        /**
         * @brief Tests whether each pair of elements is not equal.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param Xz Output array containing `1` for true comparisons and `0` otherwise.
         * @param N Number of elements to process.
         */
        static void neq(const float* __restrict Xx, const float* __restrict Xy, std::int32_t* __restrict Xz, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_COMPARE_HPP