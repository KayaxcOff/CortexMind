//
// Created by muham on 5.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Fused multiply-add operations for AVX2 vectors.
     *
     * Provides fused multiply-add and fused multiply-subtract operations
     * for single- and double-precision AVX2 vectors.
     *
     * The operations perform multiplication and addition or subtraction
     * as a single fused instruction, producing a single rounded result.
     *
     * The `nadd` and `nsub` variants negate the multiplication term before
     * performing the addition or subtraction.
     */
    struct fma {
        /**
         * @brief Computes the fused multiply-add operation.
         *
         * Computes `Xx * Xy + Xz` using a fused multiply-add instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Addend.
         *
         * @return Result of the fused multiply-add operation.
         */
        [[nodiscard]]
        static __forceinline vec8f add(const vec8f& Xx, const vec8f& Xy, const vec8f& Xz) {
            return _mm256_fmadd_ps(Xx, Xy, Xz);
        }
        /**
         * @brief Computes the fused multiply-subtract operation.
         *
         * Computes `Xx * Xy - Xz` using a fused multiply-subtract instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Subtrahend.
         *
         * @return Result of the fused multiply-subtract operation.
         */
        [[nodiscard]]
        static __forceinline vec8f sub(const vec8f& Xx, const vec8f& Xy, const vec8f& Xz) {
            return _mm256_fmsub_ps(Xx, Xy, Xz);
        }
        /**
         * @brief Computes a negated fused multiply-add operation.
         *
         * Computes `-(Xx * Xy) + Xz` using a fused negative multiply-add instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Addend.
         *
         * @return Result of the fused negative multiply-add operation.
         */
        [[nodiscard]]
        static __forceinline vec8f nadd(const vec8f& Xx, const vec8f& Xy, const vec8f& Xz) {
            return _mm256_fnmadd_ps(Xx, Xy, Xz);
        }
        /**
         * @brief Computes a negated fused multiply-subtract operation.
         *
         * Computes `-(Xx * Xy) - Xz` using a fused negative multiply-subtract instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Subtrahend.
         *
         * @return Result of the fused negative multiply-subtract operation.
         */
        [[nodiscard]]
        static __forceinline vec8f nsub(const vec8f& Xx, const vec8f& Xy, const vec8f& Xz) {
            return _mm256_fnmsub_ps(Xx, Xy, Xz);
        }

        /**
         * @brief Computes the fused multiply-add operation.
         *
         * Computes `Xx * Xy + Xz` using a fused multiply-add instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Addend.
         *
         * @return Result of the fused multiply-add operation.
         */
        [[nodiscard]]
        static __forceinline vec4d add(const vec4d& Xx, const vec4d& Xy, const vec4d& Xz) {
            return _mm256_fmadd_pd(Xx, Xy, Xz);
        }
        /**
         * @brief Computes the fused multiply-subtract operation.
         *
         * Computes `Xx * Xy - Xz` using a fused multiply-subtract instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Subtrahend.
         *
         * @return Result of the fused multiply-subtract operation.
         */
        [[nodiscard]]
        static __forceinline vec4d sub(const vec4d& Xx, const vec4d& Xy, const vec4d& Xz) {
            return _mm256_fmsub_pd(Xx, Xy, Xz);
        }
        /**
         * @brief Computes a negated fused multiply-add operation.
         *
         * Computes `-(Xx * Xy) + Xz` using a fused negative multiply-add instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Addend.
         *
         * @return Result of the fused negative multiply-add operation.
         */
        [[nodiscard]]
        static __forceinline vec4d nadd(const vec4d& Xx, const vec4d& Xy, const vec4d& Xz) {
            return _mm256_fnmadd_pd(Xx, Xy, Xz);
        }
        /**
         * @brief Computes a negated fused multiply-subtract operation.
         *
         * Computes `-(Xx * Xy) - Xz` using a fused negative multiply-subtract instruction.
         *
         * @param Xx First multiplication operand.
         * @param Xy Second multiplication operand.
         * @param Xz Subtrahend.
         *
         * @return Result of the fused negative multiply-subtract operation.
         */
        [[nodiscard]]
        static __forceinline vec4d nsub(const vec4d& Xx, const vec4d& Xy, const vec4d& Xz) {
            return _mm256_fnmsub_pd(Xx, Xy, Xz);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP