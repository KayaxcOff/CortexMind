//
// Created by muham on 8.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <cwchar>

namespace cortex::_fw::avx2 {
    struct partial {
        [[nodiscard]]
        static vec8f load(const float* src, std::size_t N);
        static void store(float* dst, const vec8f& src, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_PARTIAL_HPP