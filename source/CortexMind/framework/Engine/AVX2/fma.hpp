//
// Created by muham on 5.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>

namespace cortex::_fw::avx2 {
    struct fma {
        [[nodiscard]]
        static __forceinline vec8f add(const vec8f& x1, const vec8f& x2, const vec8f& x3) {
            return _mm256_fmadd_ps(x1, x2, x3);
        }
        [[nodiscard]]
        static __forceinline vec8f sub(const vec8f& x1, const vec8f& x2, const vec8f& x3) {
            return _mm256_fmsub_ps(x1, x2, x3);
        }
        [[nodiscard]]
        static __forceinline vec8f nadd(const vec8f& x1, const vec8f& x2, const vec8f& x3) {
            return _mm256_fnmadd_ps(x1, x2, x3);
        }
        [[nodiscard]]
        static __forceinline vec8f nsub(const vec8f& x1, const vec8f& x2, const vec8f& x3) {
            return _mm256_fnmsub_ps(x1, x2, x3);
        }

        [[nodiscard]]
        static __forceinline vec4d add(const vec4d& x1, const vec4d& x2, const vec4d& x3) {
            return _mm256_fmadd_pd(x1, x2, x3);
        }
        [[nodiscard]]
        static __forceinline vec4d sub(const vec4d& x1, const vec4d& x2, const vec4d& x3) {
            return _mm256_fmsub_pd(x1, x2, x3);
        }
        [[nodiscard]]
        static __forceinline vec4d nadd(const vec4d& x1, const vec4d& x2, const vec4d& x3) {
            return _mm256_fnmadd_pd(x1, x2, x3);
        }
        [[nodiscard]]
        static __forceinline vec4d nsub(const vec4d& x1, const vec4d& x2, const vec4d& x3) {
            return _mm256_fnmadd_pd(x1, x2, x3);
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_FMA_HPP