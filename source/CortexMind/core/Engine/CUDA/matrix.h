//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H
#define CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct Matrix {
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H