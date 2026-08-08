//
// Created by muham on 6.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/scalar.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

void ScalarOp::add(const float *Xx, const float value, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::add(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] + value;
    }
}

void ScalarOp::sub(const float *Xx, const float value, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sub(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] - value;
    }
}

void ScalarOp::mul(const float *Xx, const float value, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::mul(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * value;
    }
}

void ScalarOp::div(const float *Xx, const float value, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::div(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] / value;
    }
}

void ScalarOp::add(float *Xx, const float value, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::add(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] + value;
    }
}

void ScalarOp::sub(float *Xx, const float value, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::sub(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] - value;
    }
}

void ScalarOp::mul(float *Xx, const float value, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::mul(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * value;
    }
}

void ScalarOp::div(float *Xx, const float value, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::div(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] / value;
    }
}