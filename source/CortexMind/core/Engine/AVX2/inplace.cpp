//
// Created by muham on 13.03.2026.
//

#include "CortexMind/core/Engine/AVX2/inplace.hpp"
#include <CortexMind/core/Engine/AVX2/funcs.hpp>

using namespace cortex::_fw::avx2;

void inplace::add(f32 *Xx, const f32 *Xy, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::add(loadu(Xx + i), loadu(Xy + i));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] + Xy[i];
    }
}

void inplace::sub(f32 *Xx, const f32 *Xy, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::sub(loadu(Xx + i), loadu(Xy + i));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] - Xy[i];
    }
}

void inplace::mul(f32 *Xx, const f32 *Xy, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::mul(loadu(Xx + i), loadu(Xy + i));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] * Xy[i];
    }
}

void inplace::div(f32 *Xx, const f32 *Xy, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::div(loadu(Xx + i), loadu(Xy + i));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] / Xy[i];
    }
}

void inplace::add(f32 *Xx, const f32 value, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::add(loadu(Xx + i), set1(value));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] + value;
    }
}

void inplace::sub(f32 *Xx, const f32 value, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::sub(loadu(Xx + i), set1(value));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] - value;
    }
}

void inplace::mul(f32 *Xx, const f32 value, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::mul(loadu(Xx + i), set1(value));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] * value;
    }
}

void inplace::div(f32 *Xx, const f32 value, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::div(loadu(Xx + i), set1(value));
        storeu(Xx + i, vx);
    }
    for (; i < idx; ++i) {
        Xx[i] = Xx[i] / value;
    }
}
