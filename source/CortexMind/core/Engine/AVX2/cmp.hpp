//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    struct cmp {
        [[nodiscard]]
        static vec8f gt(vec8f Xx, vec8f Xy);
        [[nodiscard]]
        static vec8f lt(vec8f Xx, vec8f Xy);
        [[nodiscard]]
        static vec8f eq(vec8f Xx, vec8f Xy);
        [[nodiscard]]
        static vec8f ge(vec8f Xx, vec8f Xy);
        [[nodiscard]]
        static vec8f le(vec8f Xx, vec8f Xy);
        [[nodiscard]]
        static vec8f ne(vec8f Xx, vec8f Xy);
        [[nodiscard]]
        static i32 mask(vec8f x);
        [[nodiscard]]
        static bool any(vec8f x);
        [[nodiscard]]
        static bool all(vec8f x);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_CMP_HPP