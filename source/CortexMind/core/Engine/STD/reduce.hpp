//
// Created by muham on 2.05.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STD_REDUCE_HPP
#define CORTEXMIND_CORE_ENGINE_STD_REDUCE_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::stl {
    struct Reduce {
        [[nodiscard]]
        static f32 sum(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 mean(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 variance(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 standard_deviation(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 max(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 min(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 norm1(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 norm2(const f32* __restrict x, size_t N);
        [[nodiscard]]
        static f32 dot(const f32* __restrict Xx, const f32* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::stl

#endif //CORTEXMIND_CORE_ENGINE_STD_REDUCE_HPP