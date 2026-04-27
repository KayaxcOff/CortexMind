//
// Created by muham on 26.04.2026.
//

#include "CortexMind/core/Engine/STD/activation.hpp"
#include <cmath>

using namespace cortex::_fw::stl;

void ActivationOp::relu(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] > 0 ? Xx[i] : 0;
    }
}

void ActivationOp::leaky_relu(const f32 *Xx, const f32 alpha, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] > 0 ? Xx[i] : alpha * Xx[i];
    }
}

void ActivationOp::tanh(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = std::tanh(Xx[i]);
    }
}

void ActivationOp::sigmoid(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = 1.0f / (1.0f + std::exp(-Xx[i]));
    }
}

void ActivationOp::gelu(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        constexpr f32 c0 = 0.7978845608028654f;
        constexpr f32 c1 = 0.044715f;
        Xz[i] = 0.5f * Xx[i] * (1.0f + std::tanh(c0 * (Xx[i] + c1 * Xx[i] * Xx[i] * Xx[i])));
    }
}

void ActivationOp::swish(const f32 *Xx, const f32 beta, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] =  1.0f / (1.0f + std::exp(-(beta * Xx[i])));
    }
}

void ActivationOp::softmax(const f32 *Xx, f32 *Xz, size_t N) {
    f32 max_val = Xx[0];
    for (size_t i = 1; i < N; ++i) {
        if (Xx[i] > max_val) {
            max_val = Xx[i];
        }
    }

    f32 sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = std::exp(Xx[i] - max_val);
        sum += Xz[i];
    }

    for (size_t i = 0; i < N; ++i) {
        Xz[i] /= sum;
    }
}