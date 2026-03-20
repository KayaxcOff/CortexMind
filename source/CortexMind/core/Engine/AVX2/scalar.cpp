//
// Created by muham on 13.03.2026.
//

#include "CortexMind/core/Engine/AVX2/scalar.hpp"
#include <CortexMind/core/Engine/AVX2/funcs.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

void ScalarOp::add(const f32 *Xx, const f32 value, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::add(loadu(Xx + i), set1(value));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xx[i] + value;
    }
}

void ScalarOp::sub(const f32 *Xx, const f32 value, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::sub(loadu(Xx + i), set1(value));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xx[i] - value;
    }
}

void ScalarOp::mul(const f32 *Xx, const f32 value, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::mul(loadu(Xx + i), set1(value));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xx[i] * value;
    }
}

void ScalarOp::div(const f32 *Xx, const f32 value, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::div(loadu(Xx + i), set1(value));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xx[i] / value;
    }
}

f32 ScalarOp::mean(const f32 *Xx, const size_t idx) {
    vec8f acc = set_zero();
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        acc = avx2::add(acc, loadu(Xx + i));
    }
    f32 sum = hsum(acc);
    for (; i < idx; ++i) sum += Xx[i];
    return sum / static_cast<f32>(idx);
}

f32 ScalarOp::var(const f32 *Xx, const f32 mean, const size_t idx) {
    vec8f acc = set_zero();
    const vec8f vx = set1(mean);

    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f diff = avx2::sub(loadu(Xx + i), vx);
        acc = avx2::fma(diff, diff, acc);
    }
    f32 sum = hsum(acc);
    for (; i < idx; ++i) sum += (Xx[i] - mean) * (Xx[i] - mean);
    return sum / static_cast<f32>(idx);
}

f32 ScalarOp::max(const f32 *Xx, const size_t idx) {
    vec8f acc = avx2::set1(-std::numeric_limits<f32>::infinity());
    size_t i = 0;
    for (; i + 8 <= idx; i += 8)
        acc = avx2::max(acc, loadu(Xx + i));
    f32 result = hmax(acc);
    for (; i < idx; ++i) result = std::max(result, Xx[i]);
    return result;
}

f32 ScalarOp::min(const f32 *Xx, const size_t idx) {
    vec8f acc = avx2::set1(std::numeric_limits<f32>::infinity());
    size_t i = 0;
    for (; i + 8 <= idx; i += 8)
        acc = avx2::min(acc, loadu(Xx + i));
    f32 result = hmin(acc);
    for (; i < idx; ++i) result = std::min(result, Xx[i]);
    return result;
}

f32 ScalarOp::sum(const f32 *Xx, const size_t idx) {
    vec8f acc = set_zero();
    size_t i = 0;
    for (; i + 8 <= idx; i += 8)
        acc = avx2::add(acc, loadu(Xx + i));
    f32 result = hsum(acc);
    for (; i < idx; ++i) result += Xx[i];
    return result;
}

f32 ScalarOp::norm(const f32 *Xx, const size_t idx) {
    vec8f acc = set_zero();
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f v = loadu(Xx + i);
        acc = avx2::fma(v, v, acc);
    }
    f32 result = hsum(acc);
    for (; i < idx; ++i) result += Xx[i] * Xx[i];
    return std::sqrt(result);
}