//
// Created by muham on 7.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/wise.hpp"

#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <tlx/math.hpp>
#include <algorithm>
#include <cmath>

using namespace cortex::_fw::avx2;

void wise::square(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::square(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * Xx[i];
    }
}

void wise::pow(const float *Xx, const float value, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto val = set1(value);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::pow(loadu(Xx + i), val));
    }
    for (; i < N; ++i) {
        Xz[i] = std::pow(Xx[i], value);
    }
}

void wise::pow(const float *Xx, const float *Xy, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::pow(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::pow(Xx[i], Xy[i]);
    }
}

void wise::sqrt(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sqrt(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::sqrt(Xx[i]);
    }
}

void wise::rsqrt(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::rsqrt(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = 1 / std::sqrt(Xx[i]);
    }
}

void wise::log(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::log(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::log(Xx[i]);
    }
}

void wise::exp(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::exp(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::exp(Xx[i]);
    }
}

void wise::erf(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::erf(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::erf(Xx[i]);
    }
}

void wise::sin(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sin(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::sin(Xx[i]);
    }
}

void wise::cos(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::cos(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::cos(Xx[i]);
    }
}

void wise::abs(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::abs(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::abs(Xx[i]);
    }
}

void wise::neg(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::neg(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = -Xx[i];
    }
}

void wise::rcp(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::rcp(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = 1 / Xx[i];
    }
}

void wise::inverse(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::inverse(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = 1 / Xx[i];
    }
}

void wise::lerp(const float *Xx, const float value1, const float value2, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto v1 = set1(value1);
    const auto v2 = set1(value2);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::lerp(loadu(Xx + i), v1, v2));
    }
    for (; i < N; ++i) {
        Xz[i] = tlx::lerp(Xx[i], value1, value2);
    }
}

void wise::clamp(const float *Xx, const float min, const float max, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto mn = set1(min);
    const auto mx = set1(max);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::clamp(loadu(Xx + i), mn, mx));
    }
    for (; i < N; ++i) {
        Xz[i] = std::clamp(Xx[i], min, max);
    }
}

void wise::sign(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sign(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = tlx::sign(Xx[i]);
    }
}

void wise::gather(const float *Xx, const std::int32_t *Xy, float* Xz, const std::size_t N) {
    std::size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        const auto indices = loadu(Xy + i);

        storeu(Xz + i, avx2::gather(Xx, indices));
    }

    for (; i < N; ++i) {
        Xz[i] = Xx[Xy[i]];
    }
}

void wise::gather(const std::int32_t *Xx, const std::int32_t *Xy, std::int32_t* Xz, const std::size_t N) {
    std::size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        const auto indices = loadu(Xy + i);

        storeu(Xz + i, avx2::gather(Xx, indices));
    }

    for (; i < N; ++i) {
        Xz[i] = Xx[Xy[i]];
    }
}