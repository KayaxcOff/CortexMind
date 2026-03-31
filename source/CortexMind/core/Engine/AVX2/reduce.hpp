//
// Created by muham on 30.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_REDUCE_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_REDUCE_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    struct ReduceOp {
        [[nodiscard]]
        static f32 sum(const f32* Xx, size_t N);
        [[nodiscard]]
        static f32 mean(const f32* Xx, size_t N);
        [[nodiscard]]
        static f32 variance(const f32* Xx, size_t N, bool unbiased = true);
        [[nodiscard]]
        static f32 max(const f32* Xx, size_t N);
        [[nodiscard]]
        static f32 min(const f32* Xx, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_REDUCE_HPP