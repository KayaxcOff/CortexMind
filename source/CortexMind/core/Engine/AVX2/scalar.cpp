//
// Created by muham on 7.04.2026.
//

#include "CortexMind/core/Engine/AVX2/scalar.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

void ScalarOp::add(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::add(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] + value;
    }
}

void ScalarOp::sub(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sub(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] - value;
    }
}

void ScalarOp::mul(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::mul(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * value;
    }
}

void ScalarOp::div(const f32 *Xx, const f32 value, f32 *Xz, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::div(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] / value;
    }
}

void ScalarOp::add(f32 *Xx, const f32 value, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::add(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] + value;
    }
}

void ScalarOp::sub(f32 *Xx, const f32 value, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::sub(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] - value;
    }
}

void ScalarOp::mul(f32 *Xx, const f32 value, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::mul(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * value;
    }
}

void ScalarOp::div(f32 *Xx, const f32 value, const size_t N) {
    size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::div(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] / value;
    }
}