//
// Created by muham on 13.03.2026.
//

#include "CortexMind/core/Engine/AVX2/matrix.hpp"
#include <CortexMind/core/Engine/AVX2/funcs.hpp>
#include <CortexMind/core/Engine/AVX2/partial.hpp>
#include <algorithm>
#include <cmath>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

void matrix_t::add(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::add(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xy[i] + Xx[i];
    }
}

void matrix_t::sub(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::sub(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xy[i] - Xx[i];
    }
}

void matrix_t::mul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::mul(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xy[i] * Xx[i];
    }
}

void matrix_t::div(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::div(loadu(Xx + i), loadu(Xy + i));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = Xy[i] / Xx[i];
    }
}

void matrix_t::fma(const f32 *Xx, const f32 *Xy, const f32 *Xz, f32 *Xk, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::fma(loadu(Xx + i), loadu(Xy + i), loadu(Xz + i));
        storeu(Xk + i, vx);
    }
    for (; i < idx; ++i) {
        Xk[i] = Xx[i] * Xy[i] + Xz[i];
    }
}

void matrix_t::matmul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t xIdx, const size_t yIdx, const size_t zIdx) {
    constexpr size_t ROW_TILE = 4;

    for (size_t i = 0; i < xIdx; i += ROW_TILE) {
        constexpr size_t COL_TILE = 8;
        const size_t i_end = std::min(i + ROW_TILE, xIdx);

        for (size_t j = 0; j < zIdx; j += COL_TILE) {
            const size_t j_end = std::min(j + COL_TILE, zIdx);
            const size_t j_len = j_end - j;

            vec8f acc[ROW_TILE];
            for (auto & item : acc) item = set_zero();

            for (size_t k = 0; k < yIdx; ++k) {
                vec8f vy;
                if (j_len == COL_TILE) vy = loadu(Xy + k * zIdx + j);
                else                   vy = partial::load(Xy + k * zIdx + j, j_len);

                for (size_t r = 0; r < i_end - i; ++r) {
                    const vec8f vx = set1(Xx[(i + r) * yIdx + k]);
                    acc[r] = avx2::fma(vx, vy, acc[r]);
                }
            }

            for (size_t r = 0; r < i_end - i; ++r) {
                if (j_len == COL_TILE) storeu(Xz + (i + r) * zIdx + j, acc[r]);
                else                   partial::store(Xz + (i + r) * zIdx + j, acc[r], j_len);
            }
        }
    }
}

void matrix_t::sqrt(const f32 *Xx, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::sqrt(loadu(Xx + i));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = std::sqrt(Xx[i]);
    }
}

void matrix_t::pow(const f32 *Xx, const f32 value, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        const vec8f vx = avx2::pow(loadu(Xx + i), set1(value));
        storeu(Xz + i, vx);
    }
    for (; i < idx; ++i) {
        Xz[i] = std::pow(Xx[i], value);
    }
}

void matrix_t::exp(const f32 *Xx, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8)
        storeu(Xz + i, avx2::exp(loadu(Xx + i)));
    for (; i < idx; ++i) Xz[i] = std::exp(Xx[i]);
}

void matrix_t::log(const f32 *Xx, f32 *Xz, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8)
        storeu(Xz + i, avx2::log(loadu(Xx + i)));
    for (; i < idx; ++i) Xz[i] = std::log(Xx[i]);
}

void matrix_t::abs(const f32 *Xx, f32 *Xz, const size_t idx) {
    const vec8f sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    size_t i = 0;
    for (; i + 8 <= idx; i += 8)
        storeu(Xz + i, _mm256_and_ps(loadu(Xx + i), sign_mask));
    for (; i < idx; ++i) Xz[i] = std::abs(Xx[i]);
}

void matrix_t::fill(f32 *Xx, const f32 value, const size_t idx) {
    const vec8f vval = set1(value);
    size_t i = 0;
    for (; i + 8 <= idx; i += 8)
        storeu(Xx + i, vval);
    for (; i < idx; ++i)
        Xx[i] = value;
}
