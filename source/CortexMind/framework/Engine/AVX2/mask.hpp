//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>

namespace cortex::_fw::avx2 {
    [[nodiscard]]
    __forceinline vec8i mask8(const std::size_t n) {
        alignas(32) static constexpr std::int32_t mask_table[9][8] = {
            {0,  0, 0, 0, 0, 0, 0, 0},
            {-1, 0, 0, 0, 0, 0, 0, 0},
            {-1,-1, 0, 0, 0, 0, 0, 0},
            {-1,-1,-1, 0, 0, 0, 0, 0},
            {-1,-1,-1,-1, 0, 0, 0, 0},
            {-1,-1,-1,-1,-1, 0, 0, 0},
            {-1,-1,-1,-1,-1,-1, 0, 0},
            {-1,-1,-1,-1,-1,-1,-1, 0},
            {-1,-1,-1,-1,-1,-1,-1,-1}
        };

        return vec8i(mask_table[n]);
    }

    [[nodiscard]]
    __forceinline vec8i mask4(const std::size_t n) {
        alignas(32) static constexpr int64_t table[5][4] = {
            {0,  0, 0, 0},
            {-1, 0, 0, 0},
            {-1,-1, 0, 0},
            {-1,-1,-1, 0},
            {-1,-1,-1,-1}
        };

        return vec8i(_mm256_load_si256(reinterpret_cast<const __m256i*>(table[n])));
    }
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP