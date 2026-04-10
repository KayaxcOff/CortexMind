//
// Created by muham on 10.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_H
#define CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct Activation {
        static void relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void leaky_relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 alpha = 0.01f);
        static void sigmoid(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void sigmoid_fast(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void tanh(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void gelu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void gelu_exact(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void silu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void silu_fast(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void swish(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 beta = 1.0f);
        static void swish_fast(const f32* __restrict Xx, f32* __restrict Xz, size_t N, f32 beta = 1.0f);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_H