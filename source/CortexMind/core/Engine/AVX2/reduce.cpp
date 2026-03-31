//
// Created by muham on 30.03.2026.
//

#include "CortexMind/core/Engine/AVX2/reduce.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

f32 ReduceOp::sum(const f32 *Xx, const size_t N) {
    vec8f acc = zero();
    size_t i  = 0;

    for (; i + 8 <= N; i += 8){
        acc = add(acc, loadu(Xx + i));
    }

    f32 result = hsum(acc);

    for (; i < N; ++i) {
        result += Xx[i];
    }

    return result;
}

f32 ReduceOp::mean(const f32 *Xx, const size_t N) {
    if (N == 0) return 0.0f;
    return sum(Xx, N) / static_cast<f32>(N);
}

f32 ReduceOp::variance(const f32 *Xx, const size_t N, const bool unbiased) {
    if (N < 2) return 0.0f;

    f32    mean_acc = 0.0f;
    f32    m2       = 0.0f;
    size_t i        = 0;

    vec8f v_mean = zero();
    vec8f v_m2   = zero();
    vec8f v_count = zero();

    for (; i + 8 <= N; i += 8) {
        const vec8f chunk = loadu(Xx + i);
        const vec8f v_one = set1(1.0f);
        v_count = add(v_count, v_one);
        const vec8f delta  = sub(chunk, v_mean);
        v_mean             = add(v_mean, div(delta, v_count));
        const vec8f delta2 = sub(chunk, v_mean);
        v_m2               = add(v_m2, mul(delta, delta2));
    }

    alignas(32) f32 lane_mean[8];
    alignas(32) f32 lane_m2  [8];
    store(lane_mean, v_mean);
    store(lane_m2,   v_m2);

    mean_acc   = 0.0f;
    m2         = 0.0f;
    size_t cnt = 0;

    for (size_t k = 0; k < 8 && (i - (8 - k - 1)) <= N; ++k) {
        constexpr size_t cnt_b  = 1;
        const size_t cnt_ab = cnt + cnt_b;
        const f32 delta     = lane_mean[k] - mean_acc;
        mean_acc += delta * (static_cast<f32>(cnt_b) / static_cast<f32>(cnt_ab));
        m2       += lane_m2[k] + delta * (lane_mean[k] - mean_acc) * static_cast<f32>(cnt);
        cnt       = cnt_ab;
    }

    for (; i < N; ++i) {
        ++cnt;
        const f32 delta  = Xx[i] - mean_acc;
        mean_acc        += delta / static_cast<f32>(cnt);
        const f32 delta2 = Xx[i] - mean_acc;
        m2              += delta * delta2;
    }

    return m2 / static_cast<f32>(unbiased ? cnt - 1 : cnt);
}

f32 ReduceOp::max(const f32 *Xx, const size_t N) {
    if (N == 0) return 0.0f;

    vec8f acc = loadu(Xx);
    size_t i  = 8;

    for (; i + 8 <= N; i += 8)
        acc = avx2::max(acc, loadu(Xx + i));

    f32 result = hmax(acc);

    for (; i < N; ++i)
        result = Xx[i] > result ? Xx[i] : result;

    // N < 8 durumu
    if (N < 8) {
        result = Xx[0];
        for (size_t j = 1; j < N; ++j)
            result = Xx[j] > result ? Xx[j] : result;
    }

    return result;
}

f32 ReduceOp::min(const f32 *Xx, const size_t N) {
    if (N == 0) return 0.0f;

    vec8f acc = loadu(Xx);
    size_t i  = 8;

    for (; i + 8 <= N; i += 8)
        acc = avx2::min(acc, loadu(Xx + i));

    f32 result = hmin(acc);

    for (; i < N; ++i)
        result = Xx[i] < result ? Xx[i] : result;

    // N < 8 durumu
    if (N < 8) {
        result = Xx[0];
        for (size_t j = 1; j < N; ++j)
            result = Xx[j] < result ? Xx[j] : result;
    }

    return result;
}
