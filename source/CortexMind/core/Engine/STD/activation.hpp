//
// Created by muham on 26.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STD_ACTIVATION_HPP
#define CORTEXMIND_CORE_ENGINE_STD_ACTIVATION_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::stl {
    struct ActivationOp {
        static void relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void leaky_relu(const f32* __restrict Xx, f32 alpha, f32* __restrict Xz, size_t N);
        static void tanh(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void sigmoid(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void gelu(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void swish(const f32* __restrict Xx, f32 beta, f32* __restrict Xz, size_t N);
        static void softmax(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::stl

#endif //CORTEXMIND_CORE_ENGINE_STD_ACTIVATION_HPP