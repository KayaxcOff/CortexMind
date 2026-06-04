//
// Created by muham on 8.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 optimized reduction operations.
     *
     * All functions have two overloads:
     * - Scalar version: reduces entire array to a single value.
     * - Multi-dimensional version: reduces along a specific dimension.
     */
    struct reduce {
        /**
         * @brief Computes the sum of all elements in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void sum(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes sum along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void sum(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the arithmetic mean of all elements.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void mean(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes mean along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void mean(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the population variance of all elements.
         *
         * Uses the formula: `var = mean((x - μ)²)`
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void var(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes variance along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void var(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the population standard deviation.
         *
         * Equivalent to `sqrt(var(x))`
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void stdv(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes standard deviation along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void stdv(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the minimum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void min(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes minimum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void min(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the maximum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void max(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes maximum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void max(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes L1 norm (sum of absolute values).
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void norm1(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes l1 norm along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void norm1(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes L2 norm (Euclidean norm).
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         * @param N  Number of elements
         */
        static void norm2(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Computes l2 norm along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void norm2(const f32* __restrict Xx, f32* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the index of the maximum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single index)
         * @param N  Number of elements
         */
        static void argmax(const f32* __restrict Xx, i64* __restrict Xz, size_t N);
        /**
         * @brief Computes the index of maximum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void argmax(const f32* __restrict Xx, i64* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the index of the minimum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single index)
         * @param N  Number of elements
         */
        static void argmin(const f32* __restrict Xx, i64* __restrict Xz, size_t N);
        /**
         * @brief Computes the index of minimum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void argmin(const f32* __restrict Xx, i64* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the dot product of two vectors.
         *
         * @param Xx First input vector
         * @param Xy Second input vector
         * @param Xz Output array (single element)
         * @param N  Length of the vectors
         */
        static void dot(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP