//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H
#define CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct Matrix {
        static void add(const bf16* __restrict Xx, const bf16* __restrict Xy, bf16* __restrict Xz, size_t N);
        static void sub(const bf16* __restrict Xx, const bf16* __restrict Xy, bf16* __restrict Xz, size_t N);
        static void mul(const bf16* __restrict Xx, const bf16* __restrict Xy, bf16* __restrict Xz, size_t N);
        static void div(const bf16* __restrict Xx, const bf16* __restrict Xy, bf16* __restrict Xz, size_t N);

        static void add(bf16* Xx, const bf16* __restrict Xy, size_t N);
        static void sub(bf16* Xx, const bf16* __restrict Xy, size_t N);
        static void mul(bf16* Xx, const bf16* __restrict Xy, size_t N);
        static void div(bf16* Xx, const bf16* __restrict Xy, size_t N);

        static void matmul(const bf16* A, const bf16* B, bf16* C, size_t M, size_t K, size_t N, f32 alpha = 1.0f, f32 beta = 0.0f);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_MATRIX_H