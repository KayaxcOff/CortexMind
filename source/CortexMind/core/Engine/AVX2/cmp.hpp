//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief   AVX2 floating-point comparison helpers
     *
     * All comparison functions return a mask vector where bits are all-1s
     * (0xFFFFFFFF) if condition is true for that lane, and all-0s otherwise.
     */
    struct cmp {
        [[nodiscard]]
        static vec8f gt(const vec8f& x, const vec8f& y);
        [[nodiscard]]
        static vec8f lt(const vec8f& x, const vec8f& y);
        [[nodiscard]]
        static vec8f eq(const vec8f& x, const vec8f& y);
        [[nodiscard]]
        static vec8f ge(const vec8f& x, const vec8f& y);
        [[nodiscard]]
        static vec8f le(const vec8f& x, const vec8f& y);
        [[nodiscard]]
        static vec8f ne(const vec8f& x, const vec8f& y);
        [[nodiscard]]
        static i32 mask(const vec8f& x);
        [[nodiscard]]
        static bool any(const vec8f& x);
        [[nodiscard]]
        static bool all(const vec8f& x);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP