//
// Created by muham on 6.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/scalar.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

void ScalarOp::add(const float *x1, const float value, float *x2, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x2 + i, avx2::add(loadu(x1 + i), val));
    }
    for (; i < n; ++i) {
        x2[i] = x1[i] + value;
    }
}

void ScalarOp::sub(const float *x1, const float value, float *x2, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x2 + i, avx2::sub(loadu(x1 + i), val));
    }
    for (; i < n; ++i) {
        x2[i] = x1[i] - value;
    }
}

void ScalarOp::mul(const float *x1, const float value, float *x2, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x2 + i, avx2::mul(loadu(x1 + i), val));
    }
    for (; i < n; ++i) {
        x2[i] = x1[i] * value;
    }
}

void ScalarOp::div(const float *x1, const float value, float *x2, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x2 + i, avx2::div(loadu(x1 + i), val));
    }
    for (; i < n; ++i) {
        x2[i] = x1[i] / value;
    }
}

void ScalarOp::add(float *x0, const float value, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x0 + i, avx2::add(loadu(x0 + i), val));
    }
    for (; i < n; ++i) {
        x0[i] = x0[i] + value;
    }
}

void ScalarOp::sub(float *x0, const float value, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x0 + i, avx2::sub(loadu(x0 + i), val));
    }
    for (; i < n; ++i) {
        x0[i] = x0[i] - value;
    }
}

void ScalarOp::mul(float *x0, const float value, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x0 + i, avx2::mul(loadu(x0 + i), val));
    }
    for (; i < n; ++i) {
        x0[i] = x0[i] * value;
    }
}

void ScalarOp::div(float *x0, const float value, const std::size_t n) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= n; i += 8) {
        storeu(x0 + i, avx2::div(loadu(x0 + i), val));
    }
    for (; i < n; ++i) {
        x0[i] = x0[i] / value;
    }
}