//
// Created by muham on 31.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_H
#define CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_H

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct Activation {
        static void relu      (const bf16* Xx, bf16* Xz, size_t N);
        static void leaky_relu(const bf16* Xx, bf16* Xz, size_t N, f32 alpha = 0.01f);
        static void sigmoid   (const bf16* Xx, bf16* Xz, size_t N);
        static void tanh      (const bf16* Xx, bf16* Xz, size_t N);
        static void gelu      (const bf16* Xx, bf16* Xz, size_t N);
        static void silu      (const bf16* Xx, bf16* Xz, size_t N);
        static void swish     (const bf16* Xx, bf16* Xz, size_t N, f32 beta = 1.0f);
        static void softmax   (const bf16* Xx, bf16* Xz, size_t N);

        // inplace
        static void relu_   (bf16* Xx, size_t N);
        static void sigmoid_(bf16* Xx, size_t N);
        static void gelu_   (bf16* Xx, size_t N);
        static void silu_   (bf16* Xx, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_H