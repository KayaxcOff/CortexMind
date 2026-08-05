//
// Created by muham on 5.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <cstdint>

namespace cortex::_fw::avx2 {
    struct cmp {
        [[nodiscard]]
        static __forceinline vec8f gt(const vec8f& x1, const vec8f& x2) {
            return _mm256_cmp_ps(x1, x2, _CMP_GT_OQ);
        }
        [[nodiscard]]
        static __forceinline vec4d gt(const vec4d& x1, const vec4d& x2) {
            return _mm256_cmp_pd(x1, x2, _CMP_GT_OQ);
        }

        [[nodiscard]]
        static __forceinline vec8f lt(const vec8f& x1, const vec8f& x2) {
            return _mm256_cmp_ps(x1, x2, _CMP_LT_OQ);
        }
        [[nodiscard]]
        static __forceinline vec4d lt(const vec4d& x1, const vec4d& x2) {
            return _mm256_cmp_pd(x1, x2, _CMP_LT_OQ);
        }

        [[nodiscard]]
        static __forceinline vec8f eq(const vec8f& x1, const vec8f& x2) {
            return _mm256_cmp_ps(x1, x2, _CMP_EQ_OQ);
        }
        [[nodiscard]]
        static __forceinline vec4d eq(const vec4d& x1, const vec4d& x2) {
            return _mm256_cmp_pd(x1, x2, _CMP_EQ_OQ);
        }

        [[nodiscard]]
        static __forceinline vec8f ge(const vec8f& x1, const vec8f& x2) {
            return _mm256_cmp_ps(x1, x2, _CMP_GE_OQ);
        }
        [[nodiscard]]
        static __forceinline vec4d ge(const vec4d& x1, const vec4d& x2) {
            return _mm256_cmp_pd(x1, x2, _CMP_GE_OQ);
        }

        [[nodiscard]]
        static __forceinline vec8f le(const vec8f& x1, const vec8f& x2) {
            return _mm256_cmp_ps(x1, x2, _CMP_LE_OQ);
        }
        [[nodiscard]]
        static __forceinline vec4d le(const vec4d& x1, const vec4d& x2) {
            return _mm256_cmp_pd(x1, x2, _CMP_LE_OQ);
        }

        [[nodiscard]]
        static __forceinline vec8f neq(const vec8f& x1, const vec8f& x2) {
            return _mm256_cmp_ps(x1, x2, _CMP_NEQ_OQ);
        }
        [[nodiscard]]
        static __forceinline vec4d neq(const vec4d& x1, const vec4d& x2) {
            return _mm256_cmp_pd(x1, x2, _CMP_NEQ_OQ);
        }

        [[nodiscard]]
        static __forceinline std::int32_t mask(const vec8f& x) {
            return _mm256_movemask_ps(x);
        }
        [[nodiscard]]
        static __forceinline std::int32_t mask(const vec4d& x) {
            return _mm256_movemask_pd(x);
        }

        [[nodiscard]]
        static __forceinline bool any(const vec8f& x) {
            return mask(x) != 0;
        }
        [[nodiscard]]
        static __forceinline bool any(const vec4d& x) {
            return mask(x) != 0;
        }

        [[nodiscard]]
        static __forceinline bool all(const vec8f& x) {
            return mask(x) != 0xFF;
        }
        [[nodiscard]]
        static __forceinline bool all(const vec4d& x) {
            return mask(x) != 0xFF;
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_CMP_HPP