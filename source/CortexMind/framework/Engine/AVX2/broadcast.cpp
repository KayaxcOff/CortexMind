//
// Created by muham on 9.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/broadcast.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

namespace {
    template<typename OpVec, typename OpScalar>
    __forceinline void apply_row(const float* __restrict x_row, const float* __restrict y_row, float* __restrict z_row, const std::size_t N, OpVec op_vec, OpScalar op_scalar) {
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            storeu(z_row + i, op_vec(loadu(x_row + i), loadu(y_row + i)));
        }
        for (; i < N; ++i) {
            z_row[i] = op_scalar(x_row[i], y_row[i]);
        }
    }
} //unnamed namespace

void Broadcast::row::add(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::add(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] + Xy[j];
        }
    }
}

void Broadcast::row::sub(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::sub(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] - Xy[j];
        }
    }
}

void Broadcast::row::mul(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::mul(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] * Xy[j];
        }
    }
}

void Broadcast::row::div(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::mul(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] * Xy[j];
        }
    }
}

void Broadcast::row::add(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::add(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] + Xy[j];
        }
    }
}

void Broadcast::row::sub(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::sub(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] - Xy[j];
        }
    }
}

void Broadcast::row::mul(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::mul(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] * Xy[j];
        }
    }
}

void Broadcast::row::div(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::sub(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] / Xy[j];
        }
    }
}

void Broadcast::col::add(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::add(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] + val;
        }
    }
}

void Broadcast::col::sub(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::sub(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] - val;
        }
    }
}

void Broadcast::col::mul(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::mul(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] * val;
        }
    }
}

void Broadcast::col::div(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::div(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] / val;
        }
    }
}

void Broadcast::col::add(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::add(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] + val;
        }
    }
}

void Broadcast::col::sub(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::sub(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] - val;
        }
    }
}

void Broadcast::col::mul(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::mul(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] * val;
        }
    }
}

void Broadcast::col::div(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::div(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] / val;
        }
    }
}
