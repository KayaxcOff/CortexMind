//
// Created by muham on 26.04.2026.
//

#include "CortexMind/core/Engine/STD/scalar.hpp"

using namespace cortex::_fw::stl;

void Scalar::add(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] + value;
    }
}

void Scalar::sub(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] - value;
    }
}

void Scalar::mul(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] * value;
    }
}

void Scalar::div(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] / value;
    }
}

void Scalar::add(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] + value;
    }
}

void Scalar::sub(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] - value;
    }
}

void Scalar::mul(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] * value;
    }
}

void Scalar::div(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] / value;
    }
}
