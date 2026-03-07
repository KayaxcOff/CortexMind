//
// Created by muham on 21.02.2026.
//

#include "CortexMind/core/Engine/AVX2/matrix.hpp"
#include <CortexMind/core/Engine/AVX2/funcs.hpp>
#include <CortexMind/core/Engine/AVX2/partial.hpp>
#include <algorithm>

using namespace cortex::_fw::avx2;

void matrix_t::add(const f32 *x, const f32 *y, f32 *z, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        vec8f vx = loadu(x + i);
        vec8f vy = loadu(y + i);
        vec8f vz = avx2::add(vx, vy);
        storeu(z + i, vz);
    }
    for (; i < idx; ++i) z[i] = x[i] + y[i];
}

void matrix_t::sub(const f32 *x, const f32 *y, f32 *z, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        vec8f vx = loadu(x + i);
        vec8f vy = loadu(y + i);
        vec8f vz = avx2::sub(vx, vy);
        storeu(z + i, vz);
    }
    for (; i < idx; ++i) z[i] = x[i] - y[i];
}

void matrix_t::mul(const f32 *x, const f32 *y, f32 *z, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        vec8f vx = loadu(x + i);
        vec8f vy = loadu(y + i);
        vec8f vz = avx2::mul(vx, vy);
        storeu(z + i, vz);
    }
    for (; i < idx; ++i) z[i] = x[i] * y[i];
}

void matrix_t::div(const f32 *x, const f32 *y, f32 *z, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        vec8f vx = loadu(x + i);
        vec8f vy = loadu(y + i);
        vec8f vz = avx2::div(vx, vy);
        storeu(z + i, vz);
    }
    for (; i < idx; ++i) z[i] = x[i] / y[i];
}

void matrix_t::fma(const f32 *x, const f32 *y, const f32 *z, f32 *m, const size_t idx) {
    size_t i = 0;
    for (; i + 8 <= idx; i += 8) {
        vec8f vx = loadu(x + i);
        vec8f vy = loadu(y + i);
        vec8f vz = loadu(z + i);
        vec8f vm = avx2::fma(vx, vy, vz);
        storeu(m + i, vm);
    }
    for (; i < idx; ++i) m[i] = x[i] * y[i] + z[i];
}

void matrix_t::matmul(const f32 *x, const f32 *y, f32 *z, const size_t xIdx, const size_t yIdx, const size_t zIdx){
    constexpr size_t ROW_TILE = 4;

    for (size_t i = 0; i < xIdx; i += ROW_TILE) {
        constexpr size_t COL_TILE = 8;
        const size_t i_end = std::min(i + ROW_TILE, xIdx);

        for (size_t j = 0; j < zIdx; j += COL_TILE) {
            const size_t j_end = std::min(j + COL_TILE, zIdx);
            const size_t j_len = j_end - j;

            vec8f acc[ROW_TILE];
            for (auto & item : acc) item = zero();

            for (size_t k = 0; k < yIdx; ++k) {
                vec8f vy;
                if (j_len == COL_TILE) vy = loadu(y + k * zIdx + j);
                else                   vy = partial::load(y + k * zIdx + j, j_len);

                for (size_t r = 0; r < i_end - i; ++r) {
                    const vec8f vx = set1(x[(i + r) * yIdx + k]);
                    acc[r] = avx2::fma(vx, vy, acc[r]);
                }
            }

            for (size_t r = 0; r < i_end - i; ++r) {
                if (j_len == COL_TILE) storeu(z + (i + r) * zIdx + j, acc[r]);
                else                   partial::store(z + (i + r) * zIdx + j, acc[r], j_len);
            }
        }
    }
}
