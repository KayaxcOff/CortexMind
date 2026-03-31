//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    struct ScalarOp {
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);

        static void add(f32* Xx, f32 value, size_t N);
        static void sub(f32* Xx, f32 value, size_t N);
        static void mul(f32* Xx, f32 value, size_t N);
        static void div(f32* Xx, f32 value, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_SCALAR_HPP