//
// Created by muham on 7.04.2026.
//

#include "CortexMind/core/Engine/AVX2/activation.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#include <CortexMind/core/Engine/AVX2/reduce.hpp>
#include <algorithm>
#include <cmath>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

namespace {
    f32 scalar_sigmoid(const f32 x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    f32 scalar_gelu(const f32 x) {
        constexpr float c0 = 0.7978845608028654f;
        constexpr float c1 = 0.044715f;
        return 0.5f * x * (1.0f + std::tanh(c0 * (x + c1 * x * x * x)));
    }
    f32 scalar_gelu_exact(const f32 x) {
        return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
    }
} //

void Activation::relu(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::relu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::max(0.0f, Xx[i]);
    }
}

void Activation::leaky_relu(const f32* Xx, f32* Xz, const size_t N, const f32 alpha) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::leaky_relu(loadu(Xx + i), alpha));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] > 0.0f ? Xx[i] : alpha * Xx[i];
    }
}

void Activation::sigmoid(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sigmoid(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = scalar_sigmoid(Xx[i]);
    }
}

void Activation::tanh(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::tanh(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = std::tanh(Xx[i]);
    }
}

void Activation::gelu(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::gelu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = scalar_gelu(Xx[i]);
    }
}

void Activation::gelu_exact(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::gelu_exact(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = scalar_gelu_exact(Xx[i]);
    }
}

void Activation::silu(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::silu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(Xx[i]);
    }
}

void Activation::silu_fast(const f32* Xx, f32* Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::silu_fast(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(Xx[i]);
    }
}

void Activation::swish(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::swish(loadu(Xx + i), beta));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(beta * Xx[i]);
    }
}

void Activation::swish_fast(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::swish_fast(loadu(Xx + i), beta));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * scalar_sigmoid(beta * Xx[i]);
    }
}

void Activation::softmax(const f32* Xx, f32* Xz, const size_t N) {
    const vec8f vmax = set1(reduce::max(Xx, N));

    vec8f vacc = zero();
    size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        const vec8f e = avx2::exp(sub(loadu(Xx + i), vmax));
        storeu(Xz + i, e);
        vacc = add(vacc, e);
    }

    f32 sum = horizontal::sum(vacc);

    for (; i < N; ++i) {
        Xz[i] = std::exp(Xx[i] - horizontal::max(vmax));
        sum += Xz[i];
    }

    const vec8f vsum = set1(1.0f / sum);
    i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, mul(loadu(Xz + i), vsum));
    }
    for (; i < N; ++i) {
        Xz[i] *= 1.0f / sum;
    }
}

void Activation::relu(f32* Xx, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::relu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = std::max(0.0f, Xx[i]);
    }
}

void Activation::leaky_relu(f32* Xx, const size_t N, const f32 alpha) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::leaky_relu(loadu(Xx + i), alpha));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] > 0.0f ? Xx[i] : alpha * Xx[i];
    }
}

void Activation::sigmoid(f32* Xx, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::sigmoid(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = scalar_sigmoid(Xx[i]);
    }
}

void Activation::tanh(f32* Xx, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::tanh(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = std::tanh(Xx[i]);
    }
}

void Activation::gelu(f32* Xx, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::gelu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = scalar_gelu(Xx[i]);
    }
}

void Activation::gelu_exact(f32* Xx, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::gelu_exact(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = scalar_gelu_exact(Xx[i]);
    }
}

void Activation::silu(f32* Xx, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::silu(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * scalar_sigmoid(Xx[i]);
    }
}

void Activation::silu_fast(f32* Xx, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::silu_fast(loadu(Xx + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * scalar_sigmoid(Xx[i]);
    }
}

void Activation::swish(f32* Xx, const size_t N, const f32 beta) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::swish(loadu(Xx + i), beta));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * scalar_sigmoid(beta * Xx[i]);
    }
}

void Activation::swish_fast(f32* Xx, const size_t N, const f32 beta) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::swish_fast(loadu(Xx + i), beta));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * scalar_sigmoid(beta * Xx[i]);
    }
}

void Activation::softmax(f32* Xx, const size_t N) {
    f32 x_max = Xx[0];
    for (size_t i = 1; i < N; ++i) {
        x_max = std::max(x_max, Xx[i]);
    }

    const vec8f vmax = set1(x_max);
    f32 sum = 0.0f;
    size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        const vec8f e = avx2::exp(sub(loadu(Xx + i), vmax));
        storeu(Xx + i, e);
        sum += horizontal::sum(e);
    }
    for (; i < N; ++i) {
        Xx[i] = std::exp(Xx[i] - x_max);
        sum += Xx[i];
    }

    const vec8f vsum = set1(1.0f / sum);
    i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, mul(loadu(Xx + i), vsum));
    }
    for (; i < N; ++i) {
        Xx[i] *= 1.0f / sum;
    }
}