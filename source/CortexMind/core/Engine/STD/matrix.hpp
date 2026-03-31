//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::stdx {
    struct matrix {
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);

        static void add(f32* Xx, const f32* __restrict Xy, size_t N);
        static void sub(f32* Xx, const f32* __restrict Xy, size_t N);
        static void mul(f32* Xx, const f32* __restrict Xy, size_t N);
        static void div(f32* Xx, const f32* __restrict Xy, size_t N);

        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);
    };
} //namespace cortex::_fw::st

#endif //CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP