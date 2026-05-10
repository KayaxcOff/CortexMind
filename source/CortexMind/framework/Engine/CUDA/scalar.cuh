//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda {
    struct ScalarKernel {
        static void add(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void sub(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void mul(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);
        static void div(const f32* __restrict Xx, f32 value, f32* __restrict Xz, size_t N);

        static void add(f32* Xx, f32 value, size_t N);
        static void sub(f32* Xx, f32 value, size_t N);
        static void mul(f32* Xx, f32 value, size_t N);
        static void div(f32* Xx, f32 value, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH