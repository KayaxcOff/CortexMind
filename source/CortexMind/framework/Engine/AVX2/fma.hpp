//
// Created by muham on 7.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Fused Multiply-Add (FMA) operations for 8-wide float vectors.
     *
     * This struct wraps AVX2 FMA intrinsics (`_mm256_fmadd_ps`, `_mm256_fmsub_ps`, etc.).
     * All functions perform the operation in a single instruction with only one rounding.
     */
    struct fma {
        /**
         * @brief Fused Multiply-Add: `Xx * Xy + Xz`
         *
         * Computes the fused multiply-add operation in a single instruction.
         *
         * @param Xx Multiplicand vector (first factor)
         * @param Xy Multiplier vector (second factor)
         * @param Xz Addend vector (value to be added)
         * @return Result of `Xx * Xy + Xz` with single rounding
         *
         * @note Corresponds to `_mm256_fmadd_ps` intrinsic.
         * @see sub(), nadd()
         */
        [[nodiscard]]
        static __forceinline vec8f add(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
            return _mm256_fmadd_ps(Xx, Xy, Xz);
        }
        /**
         * @brief Fused Multiply-Subtract: `Xx * Xy - Xz`
         *
         * @param Xx Multiplicand vector
         * @param Xy Multiplier vector
         * @param Xz Subtrahend vector (value to be subtracted)
         * @return Result of `Xx * Xy - Xz` with single rounding
         *
         * @note Corresponds to `_mm256_fmsub_ps` intrinsic.
         */
        [[nodiscard]]
        static __forceinline vec8f sub(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
            return _mm256_fmsub_ps(Xx, Xy, Xz);
        }
        /**
         * @brief Fused Negative Multiply-Add: `-Xx * Xy + Xz`
         *
         * @param Xx Multiplicand vector (will be negated)
         * @param Xy Multiplier vector
         * @param Xz Addend vector
         * @return Result of `-Xx * Xy + Xz` with single rounding
         *
         * @note Corresponds to `_mm256_fnmadd_ps` intrinsic.
         */
        [[nodiscard]]
        static __forceinline vec8f nadd(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
            return _mm256_fnmadd_ps(Xx, Xy, Xz);
        }
        /**
         * @brief Fused Negative Multiply-Subtract: `-Xx * Xy - Xz`
         *
         * @param Xx Multiplicand vector (negated)
         * @param Xy Multiplier vector
         * @param Xz Subtrahend vector
         * @return `-Xx * Xy - Xz` computed with single rounding
         *
         * @note Corresponds to `_mm256_fnmsub_ps`
         */
        [[nodiscard]]
        static __forceinline vec8f nsub(const vec8f Xx, const vec8f Xy, const vec8f Xz) {
            return _mm256_fnmsub_ps(Xx, Xy, Xz);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP