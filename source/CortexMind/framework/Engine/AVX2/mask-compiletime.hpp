//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_COMPILETIME_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_COMPILETIME_HPP

#include <CortexMind/framework/Engine/AVX2/mask.hpp>
#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <cassert>

namespace cortex::_fw::avx2 {
    /**
     * @brief Compile-time AVX2 masked memory operations.
     *
     * Provides compile-time masked load and store operations for AVX2
     * vector types. The mask width is determined at compile time through
     * the template parameter @p N, allowing the compiler to optimize the
     * generated code for tail processing.
     *
     * The first @p N lanes are considered valid, while the remaining
     * lanes are ignored during load and store operations.
     *
     * @tparam N Number of active lanes.
     */
    template<std::size_t N>
    struct mask {
        static_assert(N >= 1 && N <= 8, "Mask size N must be in range [1, 8]");

        /**
         * @brief Performs a masked load of single-precision floating-point values.
         *
         * Loads up to @p N float values into an AVX2 vector.
         *
         * @param src Source memory.
         *
         * @return Loaded SIMD vector.
         */
        [[nodiscard]]
        static __forceinline vec8f load(const float* src) {
            return vec8f(_mm256_maskload_ps(src, mask8(N)));
        }
        /**
         * @brief Performs a masked load of 32-bit signed integers.
         *
         * Loads up to @p N integer values into an AVX2 vector.
         *
         * @param src Source memory.
         *
         * @return Loaded SIMD vector.
         */
        [[nodiscard]]
        static __forceinline vec8i load(const std::int32_t* src) {
            return vec8i(_mm256_maskload_epi32(src, mask8(N)));
        }
        /**
         * @brief Performs a masked load of double-precision floating-point values.
         *
         * Loads up to @p N double values into an AVX2 vector.
         *
         * @param src Source memory.
         *
         * @return Loaded SIMD vector.
         */
        [[nodiscard]]
        static __forceinline  vec4d load(const double* src) {
            return vec4d(_mm256_maskload_pd(src, mask4(N)));
        }

        /**
         * @brief Performs a masked store of single-precision floating-point values.
         *
         * Stores only the first @p N elements of the SIMD vector.
         *
         * @param dst Destination memory.
         * @param src Source SIMD vector.
         */
        static __forceinline void store(float* dst, const vec8f &src) {
            _mm256_maskstore_ps(dst, mask8(N), src);
        }
        /**
         * @brief Performs a masked store of 32-bit signed integers.
         *
         * Stores only the first @p N elements of the SIMD vector.
         *
         * @param dst Destination memory.
         * @param src Source SIMD vector.
         */
        static __forceinline void store(std::int32_t* dst, const vec8i &src) {
            _mm256_maskstore_epi32(dst, mask8(N), src);
        }
        /**
         * @brief Performs a masked store of double-precision floating-point values.
         *
         * Stores only the first @p N elements of the SIMD vector.
         *
         * @param dst Destination memory.
         * @param src Source SIMD vector.
         */
        static __forceinline void store(double* dst, const vec4d &src) {
            _mm256_maskstore_pd(dst, mask4(N), src);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_COMPILETIME_HPP