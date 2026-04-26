//
// Created by muham on 26.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::stl {
    struct matrix {
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);

        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);

        static void add(f32* Xx, const f32* __restrict Xy, size_t N);
        static void sub(f32* Xx, const f32* __restrict Xy, size_t N);
        static void mul(f32* Xx, const f32* __restrict Xy, size_t N);
        static void div(f32* Xx, const f32* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::stl

#endif //CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP