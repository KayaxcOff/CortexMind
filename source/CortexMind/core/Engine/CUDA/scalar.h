//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H
#define CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct ScalarKernel {
        static void add(const bf16* __restrict Xx, f32 value, bf16* __restrict Xz, size_t N);
        static void sub(const bf16* __restrict Xx, f32 value, bf16* __restrict Xz, size_t N);
        static void mul(const bf16* __restrict Xx, f32 value, bf16* __restrict Xz, size_t N);
        static void div(const bf16* __restrict Xx, f32 value, bf16* __restrict Xz, size_t N);

        static void add(bf16* Xx, f32 value, size_t N);
        static void sub(bf16* Xx, f32 value, size_t N);
        static void mul(bf16* Xx, f32 value, size_t N);
        static void div(bf16* Xx, f32 value, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H