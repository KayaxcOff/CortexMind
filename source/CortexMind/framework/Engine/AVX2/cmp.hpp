//
// Created by muham on 5.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <cstdint>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 comparison utilities.
     *
     * Provides element-wise comparison operations for AVX2 vector types
     * together with helper functions for extracting and evaluating
     * comparison masks.
     *
     * All comparison functions return SIMD mask vectors compatible with
     * subsequent AVX2 logical and masked memory operations.
     */
    struct cmp {
        /**
         * @brief Performs an element-wise greater-than comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec8f gt(const vec8f& Xx, const vec8f& Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_GT_OQ);
        }
        /**
         * @brief Performs an element-wise greater-than comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec4d gt(const vec4d& Xx, const vec4d& Xy) {
            return _mm256_cmp_pd(Xx, Xy, _CMP_GT_OQ);
        }

        /**
         * @brief Performs an element-wise less-than comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec8f lt(const vec8f& Xx, const vec8f& Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_LT_OQ);
        }
        /**
         * @brief Performs an element-wise less-than comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec4d lt(const vec4d& Xx, const vec4d& Xy) {
            return _mm256_cmp_pd(Xx, Xy, _CMP_LT_OQ);
        }

        /**
         * @brief Performs an element-wise equality comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec8f eq(const vec8f& Xx, const vec8f& Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_EQ_OQ);
        }
        /**
         * @brief Performs an element-wise inequality comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec4d eq(const vec4d& Xx, const vec4d& Xy) {
            return _mm256_cmp_pd(Xx, Xy, _CMP_EQ_OQ);
        }

        /**
         * @brief Performs an element-wise greater-than-or-equal comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec8f ge(const vec8f& Xx, const vec8f& Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_GE_OQ);
        }
        /**
         * @brief Performs an element-wise greater-than-or-equal comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec4d ge(const vec4d& Xx, const vec4d& Xy) {
            return _mm256_cmp_pd(Xx, Xy, _CMP_GE_OQ);
        }

        /**
         * @brief Performs an element-wise less-than-or-equal comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xz Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec8f le(const vec8f& Xx, const vec8f& Xz) {
            return _mm256_cmp_ps(Xx, Xz, _CMP_LE_OQ);
        }
        /**
         * @brief Performs an element-wise less-than-or-equal comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xz Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec4d le(const vec4d& Xx, const vec4d& Xz) {
            return _mm256_cmp_pd(Xx, Xz, _CMP_LE_OQ);
        }

        /**
         * @brief Performs an element-wise inequality comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec8f neq(const vec8f& Xx, const vec8f& Xy) {
            return _mm256_cmp_ps(Xx, Xy, _CMP_NEQ_OQ);
        }
        /**
         * @brief Performs an element-wise inequality comparison.
         *
         * @param Xx Left-hand operand.
         * @param Xy Right-hand operand.
         *
         * @return SIMD comparison mask.
         */
        [[nodiscard]]
        static __forceinline vec4d neq(const vec4d& Xx, const vec4d& Xy) {
            return _mm256_cmp_pd(Xx, Xy, _CMP_NEQ_OQ);
        }

        /**
         * @brief Extracts the lane mask from a SIMD comparison result.
         *
         * Returns the bit mask produced by the corresponding AVX2 movemask
         * instruction.
         *
         * @param Xx SIMD comparison vector.
         *
         * @return Lane mask.
         */
        [[nodiscard]]
        static __forceinline std::int32_t mask(const vec8f& Xx) {
            return _mm256_movemask_ps(Xx);
        }
        /**
         * @brief Extracts the lane mask from a SIMD comparison result.
         *
         * Returns the bit mask produced by the corresponding AVX2 movemask
         * instruction.
         *
         * @param Xx SIMD comparison vector.
         *
         * @return Lane mask.
         */
        [[nodiscard]]
        static __forceinline std::int32_t mask(const vec4d& Xx) {
            return _mm256_movemask_pd(Xx);
        }

        /**
         * @brief Checks whether any SIMD lane satisfies the comparison.
         *
         * @param Xx SIMD comparison vector.
         *
         * @return True if at least one lane is active.
         */
        [[nodiscard]]
        static __forceinline bool any(const vec8f& Xx) {
            return mask(Xx) != 0;
        }
        /**
         * @brief Checks whether any SIMD lane satisfies the comparison.
         *
         * @param Xx SIMD comparison vector.
         *
         * @return True if at least one lane is active.
         */
        [[nodiscard]]
        static __forceinline bool any(const vec4d& Xx) {
            return mask(Xx) != 0;
        }

        /**
         * @brief Checks whether all SIMD lanes satisfy the comparison.
         *
         * @param Xx SIMD comparison vector.
         *
         * @return True if every lane is active.
         */
        [[nodiscard]]
        static __forceinline bool all(const vec8f& Xx) {
            return mask(Xx) == 0xFF;
        }
        /**
         * @brief Checks whether all SIMD lanes satisfy the comparison.
         *
         * @param Xx SIMD comparison vector.
         *
         * @return True if every lane is active.
         */
        [[nodiscard]]
        static __forceinline bool all(const vec4d& Xx) {
            return mask(Xx) == 0xF;
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP