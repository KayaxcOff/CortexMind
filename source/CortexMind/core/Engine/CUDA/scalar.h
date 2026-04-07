//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H
#define CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct ScalarKernel {
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_H