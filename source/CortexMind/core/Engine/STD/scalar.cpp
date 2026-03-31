//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/STD/scalar.hpp"

using namespace cortex::_fw::stdx;
using namespace cortex::_fw;

void scalar_fn::add(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] + value;
    }
}

void scalar_fn::sub(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] - value;
    }
}

void scalar_fn::mul(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] * value;
    }
}

void scalar_fn::div(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] / value;
    }
}

void scalar_fn::add(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] + value;
    }
}

void scalar_fn::sub(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] - value;
    }
}

void scalar_fn::mul(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] * value;
    }
}

void scalar_fn::div(f32 *Xx, const f32 value, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] / value;
    }
}