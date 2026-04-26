//
// Created by muham on 26.04.2026.
//

#include "CortexMind/core/Engine/STD/activation.hpp"
#include <cmath>

using namespace cortex::_fw::stl;

void ActivationOp::relu(const f32 *Xx, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] > 0 ? Xz[i] : 0;
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