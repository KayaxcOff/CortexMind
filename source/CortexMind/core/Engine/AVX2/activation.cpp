//
// Created by muham on 30.03.2026.
//

#include "CortexMind/core/Engine/AVX2/activation.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#include <CortexMind/core/Engine/AVX2/partial.hpp>
#include <CortexMind/core/Engine/AVX2/reduce.hpp>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

namespace {
    template<typename Fn>
    void apply_unary(const f32* Xx, f32* Xz, const size_t N, Fn fn) {
        size_t i = 0;
        for (; i + 8 <= N; i += 8)
            storeu(Xz + i, fn(loadu(Xx + i)));
        if (i < N) {
            const vec8f tail = fn(partial::load(Xx + i, N - i));
            partial::store(Xz + i, tail, N - i);
        }
    }
} //

void Activation::relu(const f32 *Xx, f32 *Xz, const size_t N) {
    apply_unary(Xx, Xz, N, [](const vec8f v) {
        return avx2::relu(v);
    });
}

void Activation::leaky_relu(const f32 *Xx, f32 *Xz, const size_t N, f32 alpha) {
    apply_unary(Xx, Xz, N, [alpha](const vec8f v) {
        return avx2::leaky_relu(v, alpha);
    });
}

void Activation::sigmoid(const f32 *Xx, f32 *Xz, const size_t N) {
    apply_unary(Xx, Xz, N, [](const vec8f v) {
        return avx2::sigmoid(v);
    });
}

void Activation::sigmoid_fast(const f32 *Xx, f32 *Xz, const size_t N) {
    apply_unary(Xx, Xz, N, [](const vec8f v) {
        return avx2::sigmoid_fast(v);
    });
}

void Activation::tanh(const f32 *Xx, f32 *Xz, const size_t N) {
    apply_unary(Xx, Xz, N, [](const vec8f v) {
        return avx2::tanh(v);
    });
}

void Activation::gelu(const f32* Xx, f32* Xz, const size_t N) {
    apply_unary(Xx, Xz, N, [](const vec8f v) {
        return avx2::gelu(v);
    });
}

void Activation::gelu_exact(const f32* Xx, f32* Xz, const size_t N) {
    apply_unary(Xx, Xz, N, [](const vec8f v) {
        return avx2::gelu_exact(v);
    });
}

void Activation::silu(const f32* Xx, f32* Xz, const size_t N) {
    apply_unary(Xx, Xz, N, [](const vec8f v) {
        return avx2::silu(v);
    });
}

void Activation::swish(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    apply_unary(Xx, Xz, N, [beta](const vec8f v) {
        return avx2::swish(v, beta);
    });
}

void Activation::softmax(const f32* Xx, f32* Xz, const size_t N) {
    if (N == 0) return;

    const f32 x_max = ReduceOp::max(Xx, N);
    const vec8f v_max = set1(x_max);

    size_t i = 0;
    for (; i + 8 <= N; i += 8)
        storeu(Xz + i, avx2::exp(sub(loadu(Xx + i), v_max)));
    if (i < N) {
        const vec8f tail = avx2::exp(sub(partial::load(Xx + i, N - i), v_max));
        partial::store(Xz + i, tail, N - i);
    }

    const f32 total = ReduceOp::sum(Xz, N);
    const vec8f v_inv = set1(1.0f / total);

    i = 0;
    for (; i + 8 <= N; i += 8)
        storeu(Xz + i, mul(loadu(Xz + i), v_inv));
    if (i < N) {
        const vec8f tail = mul(partial::load(Xz + i, N - i), v_inv);
        partial::store(Xz + i, tail, N - i);
    }
}