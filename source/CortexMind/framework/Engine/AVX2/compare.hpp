//
// Created by muham on 6.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_COMPARE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_COMPARE_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    struct compare {
        static void gt(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N) noexcept;
        static void lt(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N) noexcept;
        static void eq(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N) noexcept;
        static void ge(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N) noexcept;
        static void le(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N) noexcept;
        static void neq(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N) noexcept;
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_COMPARE_HPP