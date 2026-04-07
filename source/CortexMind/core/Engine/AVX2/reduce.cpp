//
// Created by muham on 7.04.2026.
//

#include "CortexMind/core/Engine/AVX2/reduce.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#include <CortexMind/core/Engine/AVX2/horizontal.hpp>
#include <algorithm>
#include <cmath>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

f32 reduce::sum(const f32 *x, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = add(acc, loadu(x + i));
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += x[i];
    }
    return output;
}

f32 reduce::mean(const f32 *x, const size_t N) {
    return sum(x, N) / static_cast<f32>(N);
}

f32 reduce::var(const f32 *x, const size_t N) {
    const f32 mu = mean(x, N);
    const vec8f vmu = set1(mu);

    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        const vec8f diff = sub(loadu(x + i), vmu);
        acc = fmadd(diff, diff, acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        const f32 diff = x[i] - mu;
        output += diff * diff;
    }
    return output / static_cast<f32>(N);
}

f32 reduce::std(const f32* x, const size_t N) {
    return std::sqrt(var(x, N));
}

f32 reduce::min(const f32* x, const size_t N) {
    size_t i = 0;
    vec8f acc = set1(x[0]);
    for (; i + 8 <= N; i += 8) {
        acc = avx2::min(acc, loadu(x + i));
    }
    f32 output = horizontal::min(acc);
    for (; i < N; ++i) {
        output = std::min(output, x[i]);
    }
    return output;
}

f32 reduce::max(const f32 *x, const size_t N) {
    size_t i = 0;
    vec8f acc = set1(x[0]);
    for (; i + 8 <= N; i += 8) {
        acc = avx2::max(acc, loadu(x + i));
    }
    f32 output = horizontal::max(acc);
    for (; i < N; ++i) {
        output = std::max(output, x[i]);
    }
    return output;
}

f32 reduce::norm1(const f32 *x, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = add(acc, avx2::abs(loadu(x + i)));
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += std::abs(x[i]);
    }
    return output;
}

f32 reduce::norm2(const f32* x, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        const vec8f v = loadu(x + i);
        acc = fmadd(v, v, acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += x[i] * x[i];
    }
    return std::sqrt(output);
}

f32 reduce::dot(const f32* Xx, const f32* Xy, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = fmadd(loadu(Xx + i), loadu(Xy + i), acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += Xx[i] * Xy[i];
    }
    return output;
}