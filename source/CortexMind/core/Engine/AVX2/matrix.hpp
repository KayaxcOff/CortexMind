//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief   Collection of element-wise arithmetic operations on float arrays
     *
     * All operations are performed in-place on the output array `Xz`, using vectorized
     * AVX2 instructions for the bulk of the data and a scalar cleanup loop for the
     * remaining elements (when length is not multiple of 8).
     */
    struct matrix_t {
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise addition:  Z[i] = X[i] + Y[i]    for i = 0..idx-1
         * @param   Xx     First input array (min length `idx`)
         * @param   Xy     Second input array (min length `idx`)
         * @param   Xz     Output array (will be overwritten, min length `idx`)
         * @param   idx    Number of elements to process
         *
         * @pre     Xx, Xy, Xz point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses unaligned loads/stores → no 32-byte alignment required
         */
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise subtraction:  Z[i] = X[i] - Y[i]    for i = 0..idx-1
         * @param   Xx     Minuend array (min length `idx`)
         * @param   Xy     Subtrahend array (min length `idx`)
         * @param   Xz     Output array (will be overwritten, min length `idx`)
         * @param   idx    Number of elements to process
         *
         * @pre     Xx, Xy, Xz point to valid, disjoint memory regions of size ≥ `idx`
         */
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise multiplication:  Z[i] = X[i] × Y[i]    for i = 0..idx-1
         * @param   Xx     First factor array (min length `idx`)
         * @param   Xy     Second factor array (min length `idx`)
         * @param   Xz     Output array (will be overwritten, min length `idx`)
         * @param   idx    Number of elements to process
         *
         * @pre     Xx, Xy, Xz point to valid, disjoint memory regions of size ≥ `idx`
         */
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise fused multiply-add:  Z[i] = X[i] × Y[i] + W[i]    for i = 0..idx-1
         * @param   Xx     First input array (multiplicand)
         * @param   Xy     Second input array (multiplier)
         * @param   Xz     Array to be added
         * @param   Xk     Output array (will be overwritten, min length `idx`)
         * @param   idx    Number of elements to process
         *
         * @pre     Xx, Xy, Xz, Xk point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses `_mm256_fmadd_ps` when FMA is available; falls back to mul+add otherwise
         * @note    Uses unaligned load/store
         */
        static void fma(const f32* __restrict Xx, const f32* __restrict Xy, const f32* __restrict Xz, f32* __restrict Xk, size_t idx);
        /**
         * @brief   Matrix multiplication:  Z ← X × Y    (row-major dense matrices)
         * @param   Xx      Pointer to matrix X (row-major), shape [xIdx × yIdx]
         * @param   Xy      Pointer to matrix Y (row-major), shape [yIdx × zIdx]
         * @param   Xz      Output matrix Z (row-major), shape [xIdx × zIdx]
         * @param   xIdx   Number of rows in X and Z
         * @param   yIdx   Number of columns in X = number of rows in Y
         * @param   zIdx   Number of columns in Y and Z
         *
         * @pre     All pointers are valid and point to sufficiently large contiguous memory
         * @pre     xIdx, yIdx, zIdx > 0 (empty matrices not handled)
         * @note    Uses simple 4×8 row/column tiling + outer-product accumulation
         * @note    Partial columns handled via partial::load/store
         * @note    Performance is reasonable for small/medium sizes; not competitive with
         *          highly tuned BLAS or handwritten kernels for large matrices
         */
        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xIdx, size_t yIdx, size_t zIdx);
        /**
         * @brief   Element-wise square root: Z[i] = √X[i]    for i = 0..idx-1
         * @param   Xx     Input array
         * @param   Xz     Output array (will be overwritten)
         * @param   idx    Number of elements
         *
         * @pre     Xx, Xz point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses _mm256_sqrt_ps when available; scalar fallback for remainder
         * @note    Negative values produce NaN
         */
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise power: Z[i] = X[i]^value    for i = 0..idx-1
         * @param   Xx     Base array
         * @param   value  Exponent (scalar broadcast)
         * @param   Xz     Output array
         * @param   idx    Number of elements
         *
         * @pre     Xx, Xz point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses _mm256_pow_ps when available; scalar fallback for remainder
         * @note    Behavior for X[i] ≤ 0 depends on value
         */
        static void pow(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise exponential: Z[i] = exp(X[i])    for i = 0..idx-1
         * @param   Xx     Input array
         * @param   Xz     Output array
         * @param   idx    Number of elements
         *
         * @pre     Xx, Xz point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses _mm256_exp_ps when available; scalar fallback for remainder
         */
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise natural logarithm: Z[i] = log(X[i])    for i = 0..idx-1
         * @param   Xx     Input array (must be > 0)
         * @param   Xz     Output array
         * @param   idx    Number of elements
         *
         * @pre     Xx, Xz point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses _mm256_log_ps when available; scalar fallback for remainder
         * @note    X[i] ≤ 0 produces NaN or -Inf
         */
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t idx);
        /**
         * @brief   Element-wise absolute value: Z[i] = |X[i]|    for i = 0..idx-1
         * @param   Xx     Input array
         * @param   Xz     Output array
         * @param   idx    Number of elements
         *
         * @pre     Xx, Xz point to valid, disjoint memory regions of size ≥ `idx`
         * @note    Uses sign bit mask to clear negative sign
         */
        static void abs(const f32* __restrict Xx, f32* __restrict Xz, size_t idx);
        static void fill(f32* __restrict Xx, f32 value, size_t idx);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP