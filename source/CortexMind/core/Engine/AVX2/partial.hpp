//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    struct partial {
        [[nodiscard]]
        static vec8f load(const f32* dest, size_t N);
        static void store(f32* dest, vec8f val, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_PARTIAL_HPP