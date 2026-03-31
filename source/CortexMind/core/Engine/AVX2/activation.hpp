//
// Created by muham on 30.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_ACTIVATION_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_ACTIVATION_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    struct Activation {
        static void relu(const f32* Xx, f32* Xz, size_t N);
        static void leaky_relu(const f32* Xx, f32* Xz, size_t N, f32 alpha = 0.01f);
        static void sigmoid(const f32* Xx, f32* Xz, size_t N);
        static void sigmoid_fast(const f32* Xx, f32* Xz, size_t N);
        static void tanh(const f32* Xx, f32* Xz, size_t N);
        static void gelu (const f32* Xx, f32* Xz, size_t N);
        static void gelu_exact(const f32* Xx, f32* Xz, size_t N);
        static void silu(const f32* Xx, f32* Xz, size_t N);
        static void swish(const f32* Xx, f32* Xz, size_t N, f32 beta = 1.0f);
        static void softmax(const f32* Xx, f32* Xz, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_ACTIVATION_HPP