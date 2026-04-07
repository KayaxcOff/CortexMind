//
// Created by muham on 7.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 comparison utilities for 256-bit float vectors.
     *
     * Provides static functions for element-wise comparisons
     * and mask operations using AVX2 intrinsics.
     */
    struct cmp {
        /**
         * @brief Element-wise greater than comparison (Xx > Xy).
         * @return Vector with 0xFFFFFFFF where true, 0 otherwise.
         */
        [[nodiscard]]
        static __forceinline vec8f gt(const vec8f Xx, const vec8f Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_GT_OQ);
        }
        /**
         * @brief Element-wise less than comparison (Xx < Xy).
         * @return Vector with 0xFFFFFFFF where true, 0 otherwise.
         */
        [[nodiscard]]
        static __forceinline vec8f lt(const vec8f Xx, const vec8f Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_LT_OQ);
        }
        /**
         * @brief Element-wise equality comparison (Xx == Xy).
         * @return Vector with 0xFFFFFFFF where true, 0 otherwise.
         */
        [[nodiscard]]
        static __forceinline vec8f eq(const vec8f Xx, const vec8f Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_EQ_OQ);
        }
        /**
         * @brief Element-wise greater than or equal comparison (Xx >= Xy).
         * @return Vector with 0xFFFFFFFF where true, 0 otherwise.
         */
        [[nodiscard]]
        static __forceinline vec8f ge(const vec8f Xx, const vec8f Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_GE_OQ);
        }
        /**
         * @brief Element-wise less than or equal comparison (Xx <= Xy).
         * @return Vector with 0xFFFFFFFF where true, 0 otherwise.
         */
        [[nodiscard]]
        static __forceinline vec8f le(const vec8f Xx, const vec8f Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_LE_OQ);
        }
        /**
         * @brief Element-wise not equal comparison (Xx != Xy).
         * @return Vector with 0xFFFFFFFF where true, 0 otherwise.
         */
        [[nodiscard]]
        static __forceinline vec8f neq(const vec8f Xx, const vec8f Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_NEQ_OQ);
        }
        /**
         * @brief Converts comparison result vector to an 8-bit mask.
         * @return Integer mask where each bit represents one lane (0 or 1).
         */
        [[nodiscard]]
        static __forceinline i32 mask(const vec8f x) {
            return _mm256_movemask_ps(x);
        }
        /**
         * @brief Checks if any element in the vector is true (non-zero).
         * @return true if at least one lane is true.
         */
        [[nodiscard]]
        static __forceinline bool any(const vec8f x) {
            return mask(x) != 0;
        }
        /**
         * @brief Checks if all elements in the vector are true.
         * @return true only if all 8 lanes are true.
         */
        [[nodiscard]]
        static __forceinline bool all(const vec8f x) {
            return mask(x) == 0xFF;
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP