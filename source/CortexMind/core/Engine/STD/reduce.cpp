//
// Created by muham on 2.05.2026.
//

#include "CortexMind/core/Engine/STD/reduce.hpp"
#include <algorithm>
#include <cmath>

using namespace cortex::_fw::stl;
using namespace cortex::_fw;

f32 Reduce::sum(const f32 *x, const size_t N) {
    f32 output = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        output += x[i];
    }
    return output;
}

f32 Reduce::mean(const f32 *x, const size_t N) {
    return sum(x, N) / static_cast<f32>(N);
}

f32 Reduce::variance(const f32 *x, const size_t N) {
    const f32 mu = mean(x, N);
    f32 acc = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        const f32 diff = x[i] - mu;
        acc += diff * diff;
    }
    return acc / static_cast<f32>(N);
}

f32 Reduce::standard_deviation(const f32 *x, const size_t N) {
    return std::sqrt(variance(x, N));
}

f32 Reduce::max(const f32 *x, const size_t N) {
    f32 output = x[0];
    for (size_t i = 1; i < N; ++i) {
        output = std::max(output, x[i]);
    }
    return output;
}

f32 Reduce::min(const f32 *x, const size_t N) {
    f32 output = x[0];
    for (size_t i = 1; i < N; ++i) {
        output = std::min(output, x[i]);
    }
    return output;
}

f32 Reduce::norm1(const f32 *x, const size_t N) {
    f32 output = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        output += std::abs(x[i]);
    }
    return output;
}

f32 Reduce::norm2(const f32 *x, const size_t N) {
    f32 result = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        result += x[i] * x[i];
    }
    return std::sqrt(result);
}

f32 Reduce::dot(const f32 *Xx, const f32 *Xy, const size_t N) {
    f32 output = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        output += Xx[i] * Xy[i];
    }
    return output;
}
