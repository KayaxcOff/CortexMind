//
// Created by muham on 8.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP

#include <cstdint>
#include <cwchar>

namespace cortex::_fw::avx2 {
    /**
     * @brief SIMD-accelerated reduction operations for contiguous float data.
     *
     * Provides scalar-output reductions over a complete array and
     * dimension-wise reductions over data represented as
     * `[outer_size, dim_size, inner_size]`.
     *
     * The dimension-wise overloads reduce along the `dim_size` dimension
     * and produce an output with shape `[outer_size, inner_size]`.
     *
     * AVX2 vectorization is used for the main reduction loops, with
     * scalar processing for remaining elements that do not fill a
     * complete SIMD register.
     *
     * Supported reductions include:
     * - sum
     * - mean
     * - variance
     * - standard deviation
     * - minimum / maximum
     * - L1 / L2 norm
     * - argmin / argmax
     */
    struct reduce {
        /**
         * @brief Computes the sum of all elements.
         *
         * Computes:
         *
         * `Xz[0] = sum(Xx[i])`
         *
         * for `i` in `[0, N)`.
         *
         * @param Xx Input array.
         * @param Xz Output scalar.
         * @param N Number of elements.
         */
        static void sum(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the sum along the reduction dimension.
         *
         * Treats the input as a row-major array with shape
         * `[outer_size, dim_size, inner_size]` and reduces along
         * `dim_size`.
         *
         * The resulting output has shape `[outer_size, inner_size]`.
         *
         * @param Xx Input array with shape `[outer_size, dim_size, inner_size]`.
         * @param Xz Output array with shape `[outer_size, inner_size]`.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements along the reduction dimension.
         * @param inner_size Number of independent elements within each group.
         */
        static void sum(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Computes the arithmetic mean of all elements.
         *
         * Computes:
         *
         * `Xz[0] = sum(Xx[i]) / N`
         *
         * for `i` in `[0, N)`.
         *
         * @param Xx Input array.
         * @param Xz Output scalar.
         * @param N Number of elements.
         */
        static void mean(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the mean along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`
         * and computes the arithmetic mean along `dim_size`.
         *
         * The output has shape `[outer_size, inner_size]`.
         *
         * @param Xx Input array.
         * @param Xz Output array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being reduced.
         * @param inner_size Number of independent output elements per group.
         */
        static void mean(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Computes the population variance of all elements.
         *
         * Computes:
         *
         * `Xz[0] = (1 / N) * sum((Xx[i] - mean)^2)`
         *
         * for `i` in `[0, N)`.
         *
         * @param Xx Input array.
         * @param Xz Output scalar.
         * @param N Number of elements.
         */
        static void var(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the population variance along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`
         * and computes the variance independently for every element
         * along `inner_size`.
         *
         * The output has shape `[outer_size, inner_size]`.
         *
         * The variance is normalized by `dim_size`, corresponding to
         * population variance rather than sample variance.
         *
         * @param Xx Input array.
         * @param Xz Output array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being reduced.
         * @param inner_size Number of independent output elements per group.
         */
        static void var(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Computes the population standard deviation.
         *
         * Computes the square root of the population variance:
         *
         * `Xz[0] = sqrt(var(Xx))`
         *
         * @param Xx Input array.
         * @param Xz Output scalar.
         * @param N Number of elements.
         */
        static void stdv(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the population standard deviation along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`
         * and computes the standard deviation independently for every
         * output element.
         *
         * The output has shape `[outer_size, inner_size]`.
         *
         * @param Xx Input array.
         * @param Xz Output array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being reduced.
         * @param inner_size Number of independent output elements per group.
         */
        static void stdv(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Computes the maximum value of all elements.
         *
         * @param Xx Input array.
         * @param Xz Output scalar containing the maximum value.
         * @param N Number of elements.
         */
        static void max(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the maximum value along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`
         * and produces an output of shape `[outer_size, inner_size]`.
         *
         * @param Xx Input array.
         * @param Xz Output array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being reduced.
         * @param inner_size Number of independent output elements.
         */
        static void max(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Computes the minimum value of all elements.
         *
         * @param Xx Input array.
         * @param Xz Output scalar containing the minimum value.
         * @param N Number of elements.
         */
        static void min(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the minimum value along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`
         * and produces an output of shape `[outer_size, inner_size]`.
         *
         * @param Xx Input array.
         * @param Xz Output array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being reduced.
         * @param inner_size Number of independent output elements.
         */
        static void min(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Computes the L1 norm of all elements.
         *
         * Computes:
         *
         * `Xz[0] = sum(abs(Xx[i]))`
         *
         * for `i` in `[0, N)`.
         *
         * @param Xx Input array.
         * @param Xz Output scalar.
         * @param N Number of elements.
         */
        static void norm1(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the L1 norm along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`
         * and computes:
         *
         * `Xz[o, i] = sum(abs(Xx[o, d, i]))`
         *
         * along `d`.
         *
         * The output has shape `[outer_size, inner_size]`.
         *
         * @param Xx Input array.
         * @param Xz Output array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being reduced.
         * @param inner_size Number of independent output elements.
         */
        static void norm1(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Computes the L2 norm of all elements.
         *
         * Computes:
         *
         * `Xz[0] = sqrt(sum(Xx[i]^2))`
         *
         * for `i` in `[0, N)`.
         *
         * @param Xx Input array.
         * @param Xz Output scalar.
         * @param N Number of elements.
         */
        static void norm2(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        /**
         * @brief Computes the L2 norm along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`
         * and computes:
         *
         * `Xz[o, i] = sqrt(sum(Xx[o, d, i]^2))`
         *
         * along `d`.
         *
         * The output has shape `[outer_size, inner_size]`.
         *
         * @param Xx Input array.
         * @param Xz Output array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being reduced.
         * @param inner_size Number of independent output elements.
         */
        static void norm2(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Finds the index of the maximum element.
         *
         * Stores the zero-based index of the maximum element in `Xz[0]`.
         *
         * If multiple elements have the same maximum value, the first
         * occurrence is selected.
         *
         * @param Xx Input array.
         * @param Xz Output scalar containing the index of the maximum element.
         * @param N Number of elements.
         */
        static void argmax(const float* __restrict Xx, std::int32_t* __restrict Xz, std::size_t N);
        /**
         * @brief Finds the indices of maximum elements along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`.
         *
         * For every `[outer, inner]` position, finds the index along
         * `dim_size` containing the maximum value.
         *
         * The output has shape `[outer_size, inner_size]`.
         *
         * Indices are zero-based and refer to positions within the
         * reduction dimension.
         *
         * @param Xx Input array.
         * @param Xz Output index array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being searched.
         * @param inner_size Number of independent output elements.
         */
        static void argmax(const float* __restrict Xx, std::int32_t* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        /**
         * @brief Finds the index of the minimum element.
         *
         * Stores the zero-based index of the minimum element in `Xz[0]`.
         *
         * If multiple elements have the same minimum value, the first
         * occurrence is selected.
         *
         * @param Xx Input array.
         * @param Xz Output scalar containing the index of the minimum element.
         * @param N Number of elements.
         */
        static void argmin(const float* __restrict Xx, std::int32_t* __restrict Xz, std::size_t N);
        /**
         * @brief Finds the indices of maximum elements along the reduction dimension.
         *
         * Treats the input as `[outer_size, dim_size, inner_size]`.
         *
         * For every `[outer, inner]` position, finds the index along
         * `dim_size` containing the maximum value.
         *
         * The output has shape `[outer_size, inner_size]`.
         *
         * Indices are zero-based and refer to positions within the
         * reduction dimension.
         *
         * @param Xx Input array.
         * @param Xz Output index array.
         * @param outer_size Number of outer groups.
         * @param dim_size Number of elements being searched.
         * @param inner_size Number of independent output elements.
         */
        static void argmin(const float* __restrict Xx, std::int32_t* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP