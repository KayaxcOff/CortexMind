//
// Created by muham on 7.04.2026.
//

#include "CortexMind/core/Engine/AVX2/matrix.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>
#include <CortexMind/core/Engine/AVX2/mask.hpp>
#include <algorithm>

using namespace cortex::_fw::avx2;

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

void matrix_t::fmadd(const f32 *Xx, const f32 *Xy, const f32 *Xz, f32 *Xk, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xk + i, avx2::fmadd(loadu(Xx + i), loadu(Xy + i), loadu(Xz + i)));
    }
    for (; i < N; ++i) {
        Xk[i] = Xx[i] * Xy[i] + Xz[i];
    }
}

void matrix_t::fmsub(const f32 *Xx, const f32 *Xy, const f32 *Xz, f32 *Xk, const size_t N) {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xk + i, avx2::fmsub(loadu(Xx + i), loadu(Xy + i), loadu(Xz + i)));
    }
    for (; i < N; ++i) {
        Xk[i] = Xx[i] * Xy[i] - Xz[i];
    }
}

void matrix_t::matmul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t xN, const size_t yN, const size_t zN) {
    constexpr size_t ROW_TILE = 4;
    constexpr size_t TILE_K = 64;
    constexpr size_t COL_TILE = 8;

    for (size_t i = 0; i < xN; i += ROW_TILE) {
        const size_t i_len = std::min(ROW_TILE, xN - i);

        for (size_t j = 0; j < zN; j += COL_TILE) {
            const size_t j_len = std::min(COL_TILE, zN - j);

            const mask col_mask(j_len);
            if (i_len == ROW_TILE && j_len == COL_TILE) {
                vec8f acc0 = zero(), acc1 = zero(), acc2 = zero(), acc3 = zero();

                for (size_t k0 = 0; k0 < yN; k0 += TILE_K) {
                    const size_t k_end = std::min(k0 + TILE_K, yN);
                    for (size_t k = k0; k < k_end; ++k) {
                        const vec8f vy = loadu(Xy + k * zN + j);

                        acc0 = avx2::fmadd(set1(Xx[(i + 0) * yN + k]), vy, acc0);
                        acc1 = avx2::fmadd(set1(Xx[(i + 1) * yN + k]), vy, acc1);
                        acc2 = avx2::fmadd(set1(Xx[(i + 2) * yN + k]), vy, acc2);
                        acc3 = avx2::fmadd(set1(Xx[(i + 3) * yN + k]), vy, acc3);
                    }
                }
                storeu(Xz + (i + 0) * zN + j, acc0);
                storeu(Xz + (i + 1) * zN + j, acc1);
                storeu(Xz + (i + 2) * zN + j, acc2);
                storeu(Xz + (i + 3) * zN + j, acc3);
            }
            else {
                vec8f acc[ROW_TILE];
                for (auto& item : acc) item = zero();

                for (size_t k = 0; k < yN; ++k) {
                    const vec8f vy = (j_len == COL_TILE) ? loadu(Xy + k * zN + j) : col_mask.load(Xy + k * zN + j);

                    for (size_t r = 0; r < i_len; ++r) {
                        acc[r] = avx2::fmadd(set1(Xx[(i + r) * yN + k]), vy, acc[r]);
                    }
                }

                for (size_t r = 0; r < i_len; ++r) {
                    if (j_len == COL_TILE) storeu(Xz + (i + r) * zN + j, acc[r]);
                    else col_mask.store(Xz + (i + r) * zN + j, acc[r]);
                }
            }
        }
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