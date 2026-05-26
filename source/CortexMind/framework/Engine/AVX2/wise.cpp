//
// Created by muham on 8.05.2026.
//

#include "CortexMind/framework/Engine/AVX2/wise.hpp"
#include <CortexMind/framework/Engine/AVX2/cmp.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace cortex::_fw::avx2;

void wise::pow(const f32 *Xx, const f32 exp, f32 *Xz, const size_t N) {
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
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sqrt(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::sqrt(Xx[i]);
    }
}

void wise::rsqrt(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::rsqrt(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = 1 / std::sqrt(Xx[i]);
    }
}

void wise::log(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::log(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::log(Xx[i]);
    }
}

void wise::exp(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::exp(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::exp(Xx[i]);
    }
}

void wise::sin(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sin(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::sin(Xx[i]);
    }
}

void wise::cos(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::cos(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::cos(Xx[i]);
    }
}

void wise::abs(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::abs(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::abs(Xx[i]);
    }
}

void wise::neg(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::neg(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = -std::abs(Xx[i]);
    }
}

void wise::sign(const f32 *Xx, f32 *Xz, const size_t N) {
    size_t i = 0;

    const auto zeros = zero();
    const auto one = set1(1.0f);

    for (; i + 8 <= N; i += 8) {
        const vec8f x = loadu(Xx + i);

        const vec8f gt_mask = cmp::gt(x, zeros);
        const vec8f lt_mask = cmp::lt(x, zeros);

        const vec8f pos_vals = _mm256_and_ps(gt_mask, one);
        const vec8f neg_vals = _mm256_and_ps(lt_mask, one);

        const vec8f res = sub(pos_vals, neg_vals);

        storeu(Xz + i, res);
    }
    for (; i < N; ++i) {
        const f32 val = Xx[i];
        Xz[i] = static_cast<f32>((0.0f < val) - (val < 0.0f));
    }
}

void wise::greater(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    size_t i = 0;
    const vec8f one = set1(1.0f);
    for (; i + 8 <= N; i += 8) {
        const vec8f mask = cmp::gt(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, _mm256_and_ps(mask, one));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] > Xy[i] ? 1.0f : 0.0f;
    }
}

void wise::less(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    size_t i = 0;
    const vec8f one = set1(1.0f);
    for (; i + 8 <= N; i += 8) {
        const vec8f mask = cmp::lt(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, _mm256_and_ps(mask, one));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] < Xy[i] ? 1.0f : 0.0f;
    }
}

void wise::greater_eq(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    size_t i = 0;
    const vec8f one = set1(1.0f);
    for (; i + 8 <= N; i += 8) {
        const vec8f mask = cmp::ge(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, _mm256_and_ps(mask, one));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] >= Xy[i] ? 1.0f : 0.0f;
    }
}

void wise::less_eq(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    size_t i = 0;
    const vec8f one = set1(1.0f);
    for (; i + 8 <= N; i += 8) {
        const vec8f mask = cmp::le(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, _mm256_and_ps(mask, one));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] <= Xy[i] ? 1.0f : 0.0f;
    }
}

void wise::clamp(const f32 *Xx, const f32 min_val, const f32 max_val, f32 *Xz, const size_t N) {
    size_t i = 0;
    const vec8f vmin = set1(min_val);
    const vec8f vmax = set1(max_val);
    for (; i + 8 <= N; i += 8) {
        const vec8f x = loadu(Xx + i);
        storeu(Xz + i, min(max(x, vmin), vmax));
    }
    for (; i < N; ++i) {
        Xz[i] = std::max(min_val, std::min(Xx[i], max_val));
    }
}