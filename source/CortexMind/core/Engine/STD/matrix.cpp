//
// Created by muham on 26.04.2026.
//

#include "CortexMind/core/Engine/STD/matrix.hpp"

using namespace cortex::_fw::stl;

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

void matrix::matmul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t xN, const size_t yN, const size_t zN) {
    for (size_t i = 0; i < xN; ++i) {
        for (size_t k = 0; k < zN; ++k) {
            f32 sum = 0.0f;

            for (size_t j = 0; j < yN; ++j) {
                sum += Xx[i * yN + j] * Xy[j * zN + k];
            }

            Xz[i * zN + k] = sum;
        }
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