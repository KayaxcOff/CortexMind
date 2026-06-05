//
// Created by muham on 5.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_REDUCE_REDUCE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_REDUCE_REDUCE_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    struct TensorReduce {
        /**
         * @brief Computes the sum of all elements in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void sum(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes sum along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void sum(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the arithmetic mean of all elements.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void mean(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes mean along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void mean(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the population variance of all elements.
         *
         * Uses the formula: `var = mean((x - μ)²)`
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void var(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes variance along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void var(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the population standard deviation.
         *
         * Equivalent to `sqrt(var(x))`
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void stdv(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes standard deviation along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void stdv(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the minimum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void min(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes minimum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void min(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the maximum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void max(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes maximum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void max(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes L1 norm (sum of absolute values).
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void norm1(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes l1 norm along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void norm1(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes L2 norm (Euclidean norm).
         *
         * @param Xx Input array
         * @param Xz Output array (single element)
         */
        static void norm2(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        /**
         * @brief Computes l2 norm along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void norm2(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the index of the maximum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single index)
         */
        static void argmax(const TensorStorage* __restrict Xx, i64* __restrict Xz);
        /**
         * @brief Computes the index of maximum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void argmax(const TensorStorage* __restrict Xx, i64* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Finds the index of the minimum value in the array.
         *
         * @param Xx Input array
         * @param Xz Output array (single index)
         */
        static void argmin(const TensorStorage* __restrict Xx, i64* __restrict Xz);
        /**
         * @brief Computes the index of minimum value along a specified dimension.
         *
         * @param Xx          Input array
         * @param Xz          Output array
         * @param outer_size  Size of dimensions before the reduced one
         * @param dim_size    Size of the dimension being reduced
         * @param inner_size  Size of dimensions after the reduced one
         */
        static void argmin(const TensorStorage* __restrict Xx, i64* __restrict Xz, size_t outer_size, size_t dim_size, size_t inner_size);
        /**
         * @brief Computes the dot product of two vectors.
         *
         * @param Xx First input vector
         * @param Xy Second input vector
         * @param Xz Output array (single element)
         */
        static void dot(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_REDUCE_REDUCE_HPP