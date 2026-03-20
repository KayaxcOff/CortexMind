//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief   Collection of element-wise operations between a vector and a scalar
     *
     * Each function computes  Z[i] = X[i]  <op>  value   for i = 0..idx-1
     * using AVX2 vectorized code + scalar remainder loop.
     */
    struct ScalarOp {
        /**
         * @brief   Z[i] = X[i] + value    for i = 0..idx-1
         * @param   Xx     Input array (source) – min length `idx`
         * @param   value  Scalar value to add to each element
         * @param   Xz     Output array (destination) – will be overwritten, min length `idx`
         * @param   idx    Number of elements to process
         *
         * @pre     Xx and Xz point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses unaligned loads/stores
         */
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] - value    for i = 0..idx-1
         * @param   Xx     Input array (minuend)
         * @param   value  Scalar value to subtract from each element
         * @param   Xz     Output array (destination)
         * @param   idx    Number of elements to process
         */
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] × value    for i = 0..idx-1
         * @param   Xx     Input array (first factor)
         * @param   value  Scalar multiplier
         * @param   Xz     Output array (destination)
         * @param   idx    Number of elements to process
         */
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Z[i] = X[i] ÷ value    for i = 0..idx-1
         * @param   Xx     Input array (dividend)
         * @param   value  Scalar divisor (should be non-zero)
         * @param   Xz     Output array (destination)
         * @param   idx    Number of elements to process
         *
         * @warning Division by zero → Inf or NaN in all elements
         */
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Computes the arithmetic mean of the first `idx` elements
         * @param   Xx     Input array (min length `idx`)
         * @param   idx    Number of elements to average
         * @return  Mean value = (sum(X[0..idx-1])) / idx
         *
         * @pre     idx > 0
         * @pre     Xx points to valid memory of at least `idx` floats
         * @note    Uses horizontal sum + scalar cleanup loop
         * @note    Result is returned as `f32` (may lose precision for very large sums)
         */
        static f32 mean(const f32* __restrict Xx, size_t idx);
        /**
         * @brief   Computes variance of the first `idx` elements
         * @param   Xx     Input array (min length `idx`)
         * @param   mean   Pre-computed mean value (caller responsibility)
         * @param   idx    Number of elements
         * @return  Variance = (sum((X[i] - mean)²)) / idx
         *
         * @pre     idx > 0
         * @note    Unbiased estimator (divides by idx, not idx-1)
         * @note    Uses FMA for better numerical stability
         */
        static f32 var(const f32* __restrict Xx, f32 mean, size_t idx);
        /**
         * @brief   Computes maximum value in the array
         * @param   Xx     Input array (min length `idx`)
         * @param   idx    Number of elements
         * @return  Maximum value among X[0..idx-1]
         *
         * @pre     idx > 0
         * @note    Uses horizontal max + scalar cleanup
         */
        static f32 max(const f32* __restrict Xx, size_t idx);
        /**
         * @brief   Computes minimum value in the array
         * @param   Xx     Input array (min length `idx`)
         * @param   idx    Number of elements
         * @return  Minimum value among X[0..idx-1]
         */
        static f32 min(const f32* __restrict Xx, size_t idx);
        /**
         * @brief   Computes Euclidean (L2) norm of the array
         * @param   Xx     Input array (min length `idx`)
         * @param   idx    Number of elements
         * @return  √(sum(X[i]²)) for i = 0..idx-1
         *
         * @pre     idx > 0
         * @note    Uses FMA for inner product accumulation
         * @note    Scalar cleanup for remainder
         */
        static f32 norm(const f32* __restrict Xx, size_t idx);
        /**
         * @brief   Computes sum of all elements
         * @param   Xx     Input array (min length `idx`)
         * @param   idx    Number of elements
         * @return  Sum of X[0..idx-1]
         *
         * @pre     idx > 0
         * @note    Uses horizontal sum + scalar cleanup
         */
        static f32 sum(const f32* __restrict Xx, size_t idx);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP