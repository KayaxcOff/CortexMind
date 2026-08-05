//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP

#include <CortexMind/framework/Engine/AVX2/types.hpp>
#include <cstdint>
#include <cstddef>

namespace cortex::_fw::avx2 {
    /**
     * @brief Creates an 8-lane AVX2 mask.
     *
     * Returns a mask vector containing the first @p n lanes enabled and
     * the remaining lanes disabled. Enabled lanes are represented by
     * all bits set (`-1`), while disabled lanes contain zero.
     *
     * The returned mask is intended for AVX2 masked load/store operations
     * such as `_mm256_maskload_ps()` and `_mm256_maskstore_ps()`.
     *
     * @param n Number of active lanes in the range [0, 8].
     *
     * @return AVX2 mask vector with @p n active lanes.
     */
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

        return _mm256_load_si256(reinterpret_cast<const vec8i*>(mask_table[n]));
    }

    /**
     * @brief Creates a 4-lane AVX2 mask.
     *
     * Returns a mask vector containing the first @p n 64-bit lanes enabled
     * and the remaining lanes disabled. Enabled lanes are represented by
     * all bits set (`-1`), while disabled lanes contain zero.
     *
     * The returned mask is intended for AVX2 masked operations working on
     * four 64-bit elements.
     *
     * @param n Number of active lanes in the range [0, 4].
     *
     * @return AVX2 mask vector with @p n active lanes.
     */
    [[nodiscard]]
    __forceinline vec8i mask4(const std::size_t n) {
        alignas(32) static constexpr int64_t table[5][4] = {
            {0,  0, 0, 0},
            {-1, 0, 0, 0},
            {-1,-1, 0, 0},
            {-1,-1,-1, 0},
            {-1,-1,-1,-1}
        };

        return _mm256_load_si256(reinterpret_cast<const vec8i*>(table[n]));
    }
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MASK_HPP