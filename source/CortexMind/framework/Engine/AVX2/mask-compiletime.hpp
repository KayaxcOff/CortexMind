//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_COMPILETIME_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_COMPILETIME_HPP

#include <CortexMind/framework/Engine/AVX2/mask.hpp>
#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <cassert>

namespace cortex::_fw::avx2 {
    template<std::size_t N>
    struct mask {
        static_assert(N >= 1 && N <= 8, "Mask size N must be in range [1, 8]");

        [[nodiscard]]
        static __forceinline vec8f load(const float* src) {
            return vec8f(_mm256_maskload_ps(src, Init(N).raw()));
        }
        static __forceinline void store(float* dst, const vec8f &src) {
            _mm256_maskstore_ps(dst, Init(N).raw(), src.raw());
        }
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_COMPILETIME_HPP