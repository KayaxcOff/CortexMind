//
// Created by muham on 7.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief SIMD comparison operations for 8-wide float vectors (AVX2).
     *
     * This struct contains static utility functions that wrap AVX2 comparison
     * intrinsics (`_mm256_cmp_ps`). All functions return a mask vector where
     * each bit field corresponds to the comparison result for that lane.
     */
    struct cmp {
        /**
         * @brief Performs greater-than comparison (X > Y) on two __m256 vectors.
         *
         * @param Xx Left-hand side vector (first operand).
         * @param Xy Right-hand side vector (second operand).
         * @return A mask vector (__m256) where each 32-bit lane is 0xFFFFFFFF if
         *         the corresponding element in Xx is greater than in Xy, otherwise 0.
         *
         * @note Uses `_CMP_GT_OQ` (Ordered, Quiet) comparison.
         * @see lt(), ge(), _mm256_cmp_ps
         */
        [[nodiscard]]
        static __forceinline vec8f gt(const vec8f Xx, const vec8f Xy) noexcept {
            return _mm256_cmp_ps(Xx, Xy, _CMP_GT_OQ);
        }
        /**
         * @brief Performs less-than comparison (X < Y) on two __m256 vectors.
         *
         * @param Xx Left-hand side vector.
         * @param Xy Right-hand side vector.
         * @return Mask vector with 0xFFFFFFFF where Xx < Xy.
         *
         * @note Uses `_CMP_LT_OQ` (Ordered, Quiet) comparison.
         */
        [[nodiscard]]
        static __forceinline vec8f lt(const vec8f Xx, const vec8f Xy) noexcept {
            return _mm256_cmp_ps(Xx, Xy, _CMP_LT_OQ);
        }
        /**
         * @brief Performs equality comparison (X == Y) on two __m256 vectors.
         *
         * @param Xx Left-hand side vector.
         * @param Xy Right-hand side vector.
         * @return Mask vector with 0xFFFFFFFF where elements are equal.
         *
         * @note Uses `_CMP_EQ_OQ` (Ordered, Quiet) comparison.
         * @warning Due to floating-point precision, exact equality should be used
         *          with caution. Consider using a small epsilon when appropriate.
         */
        [[nodiscard]]
        static __forceinline vec8f eq(const vec8f Xx, const vec8f Xy) noexcept {
            return _mm256_cmp_ps(Xx, Xy, _CMP_EQ_OQ);
        }
        /**
         * @brief Performs greater-than-or-equal comparison (X >= Y).
         *
         * @param Xx Left-hand side vector.
         * @param Xy Right-hand side vector.
         * @return Mask vector with 0xFFFFFFFF where Xx >= Xy.
         *
         * @note Uses `_CMP_GE_OQ` (Ordered, Quiet) comparison.
         */
        [[nodiscard]]
        static __forceinline vec8f ge(const vec8f Xx, const vec8f Xy) noexcept {
            return _mm256_cmp_ps(Xx, Xy, _CMP_GE_OQ);
        }
        /**
         * @brief Performs less-than-or-equal comparison (X <= Y).
         *
         * @param Xx Left-hand side vector.
         * @param Xy Right-hand side vector.
         * @return Mask vector with 0xFFFFFFFF where Xx <= Xy.
         *
         * @note Uses `_CMP_LE_OQ` (Ordered, Quiet) comparison.
         */
        [[nodiscard]]
        static __forceinline vec8f le(const vec8f Xx, const vec8f Xy) noexcept {
            return _mm256_cmp_ps(Xx, Xy, _CMP_LE_OQ);
        }
        /**
         * @brief Performs not-equal comparison (X != Y).
         *
         * @param Xx Left-hand side vector.
         * @param Xy Right-hand side vector.
         * @return Mask vector with 0xFFFFFFFF where elements are not equal.
         *
         * @note Uses `_CMP_NEQ_OQ` (Ordered, Quiet) comparison.
         */
        [[nodiscard]]
        static __forceinline vec8f neq(const vec8f Xx, const vec8f Xy) noexcept {
            return _mm256_cmp_ps(Xx, Xy, _CMP_NEQ_OQ);
        }
        /**
         * @brief Extracts the comparison mask as a 32-bit integer.
         *
         * Converts the most significant bit of each 32-bit lane in the input
         * vector into the corresponding bit in the returned integer.
         *
         * @param x The comparison result vector (usually from gt/lt/eq etc.).
         * @return Integer mask (bits 0-7 represent the 8 lanes).
         *
         * @see any(), all(), _mm256_movemask_ps
         */
        [[nodiscard]]
        static __forceinline i32 mask(const vec8f x) noexcept {
            return _mm256_movemask_ps(x);
        }
        /**
         * @brief Checks if any element in the comparison mask is true.
         *
         * @param x The comparison result vector.
         * @return `true` if at least one lane satisfies the comparison condition.
         *
         * @note Equivalent to `mask(x) != 0`
         */
        [[nodiscard]]
        static __forceinline bool any(const vec8f x) noexcept {
            return mask(x) != 0;
        }
        /**
         * @brief Checks if all elements in the comparison mask are true.
         *
         * @param x The comparison result vector.
         * @return `true` only if all 8 lanes satisfy the comparison condition.
         *
         * @note Equivalent to `mask(x) == 0xFF`
         */
        [[nodiscard]]
        static __forceinline bool all(const vec8f x) noexcept {
            return mask(x) == 0xFF;
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP