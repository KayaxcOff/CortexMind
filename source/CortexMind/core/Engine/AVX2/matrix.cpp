//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/AVX2/matrix.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#include <CortexMind/core/Engine/AVX2/partial.hpp>
#include <algorithm>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

void matrix_t::add(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::add(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] + Xy[i];
    }
}

void matrix_t::sub(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sub(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] - Xy[i];
    }
}

void matrix_t::mul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::mul(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * Xy[i];
    }
}

void matrix_t::div(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::div(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] / Xy[i];
    }
}

void matrix_t::add(f32 *Xx, const f32 *Xy, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::add(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] + Xy[i];
    }
}

void matrix_t::sub(f32 *Xx, const f32 *Xy, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::sub(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] - Xy[i];
    }
}

void matrix_t::mul(f32 *Xx, const f32 *Xy, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::mul(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * Xy[i];
    }
}

void matrix_t::div(f32 *Xx, const f32 *Xy, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::div(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] / Xy[i];
    }
}

void matrix_t::fma(const f32 *Xx, const f32 *Xy, const f32 *Xz, f32 *Xk, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f vx = avx2::fma(loadu(Xx + i), loadu(Xy + i), loadu(Xz + i));
        storeu(Xk + i, vx);
    }
    for (; i < N; ++i) {
        Xk[i] = Xx[i] * Xy[i] + Xz[i];
    }
}

void matrix_t::matmul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t xN, const size_t yN, const size_t zN) {
    constexpr size_t ROW_TILE = 4;

    for (size_t i = 0; i < xN; i += ROW_TILE) {
        constexpr size_t COL_TILE = 8;
        const size_t i_end = std::min(i + ROW_TILE, xN);

        for (size_t j = 0; j < zN; j += COL_TILE) {
            const size_t j_end = std::min(j + COL_TILE, zN);
            const size_t j_len = j_end - j;

            vec8f acc[ROW_TILE];
            for (auto & item : acc) item = zero();

            for (size_t k = 0; k < yN; ++k) {
                vec8f vy;
                if (j_len == COL_TILE) vy = loadu(Xy + k * zN + j);
                else                   vy = partial::load(Xy + k * zN + j, j_len);

                for (size_t r = 0; r < i_end - i; ++r) {
                    const vec8f vx = set1(Xx[(i + r) * yN + k]);
                    acc[r] = avx2::fma(vx, vy, acc[r]);
                }
            }

            for (size_t r = 0; r < i_end - i; ++r) {
                if (j_len == COL_TILE) storeu(Xz + (i + r) * zN + j, acc[r]);
                else                   partial::store(Xz + (i + r) * zN + j, acc[r], j_len);
            }
        }
    }
}