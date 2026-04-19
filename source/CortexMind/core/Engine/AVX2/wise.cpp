//
// Created by muham on 19.04.2026.
//

#include "CortexMind/core/Engine/AVX2/wise.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#include <cmath>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

void wise::pow(const f32 *Xx, const f32 exp, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::pow(loadu(Xx + i), set1(exp)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::pow(Xz[i], exp);
    }
}

void wise::square(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, sqrt(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::sqrt(Xz[i]);
    }
}

void wise::log(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::log(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::log(Xz[i]);
    }
}

void wise::exp(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::exp(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::exp(Xz[i]);
    }
}
