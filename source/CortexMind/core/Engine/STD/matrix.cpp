//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/STD/matrix.hpp"
#include <cstring>

using namespace cortex::_fw::stdx;
using namespace cortex::_fw;

void matrix::add(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] + Xy[i];
    }
}

void matrix::sub(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] - Xy[i];
    }
}

void matrix::mul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] * Xy[i];
    }
}

void matrix::div(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xz[i] = Xx[i] / Xy[i];
    }
}

void matrix::add(f32 *Xx, const f32 *Xy, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] + Xy[i];
    }
}

void matrix::sub(f32 *Xx, const f32 *Xy, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] - Xy[i];
    }
}

void matrix::mul(f32 *Xx, const f32 *Xy, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] * Xy[i];
    }
}

void matrix::div(f32 *Xx, const f32 *Xy, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Xx[i] = Xx[i] / Xy[i];
    }
}

void matrix::matmul(const f32 *Xx, const f32 *Xy, f32* Xz, const size_t xN, const size_t yN, const size_t zN) {

    std::memset(Xz, 0, xN * zN * sizeof(f32));
    for (size_t i = 0; i < xN; ++i) {
        for (size_t k = 0; k < yN; ++k) {
            const f32 a = Xx[i * yN + k];
            for (size_t j = 0; j < zN; ++j) {
                Xz[i * zN + j] += a * Xy[k * zN + j];
            }
        }
    }
}
