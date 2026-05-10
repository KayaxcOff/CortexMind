//
// Created by muham on 8.05.2026.
//

#include "CortexMind/framework/Engine/AVX2/wise.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <cmath>
#include <cstdlib>

using namespace cortex::_fw::avx2;

void wise::pow(const f32 *Xx, const f32 exp, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    const auto val = set1(exp);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::pow(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = std::pow(Xx[i], exp);
    }
}

void wise::sqrt(const f32 *Xx, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sqrt(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::sqrt(Xx[i]);
    }
}

void wise::log(const f32 *Xx, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::log(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::log(Xx[i]);
    }
}

void wise::exp(const f32 *Xx, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::exp(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::exp(Xx[i]);
    }
}

void wise::sin(const f32 *Xx, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sin(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::sin(Xx[i]);
    }
}

void wise::abs(const f32 *Xx, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::abs(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::abs(Xx[i]);
    }
}
