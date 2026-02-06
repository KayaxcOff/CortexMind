//
// Created by muham on 2.02.2026.
//

#include "CortexMind/core/Engine/AVX/matrix.hpp"
#include <CortexMind/core/Engine/AVX/partial.hpp>
#include <algorithm>

using namespace cortex::_fw::avx2;

void matrix_t::add(const f32 *_x, const f32 *_y, f32 *_z, const size_t idx) {
    size_t i = 0;

    for (; i + 8 < idx; i += 8) {
        const vec8f vx = loadu(_x + i);
        const vec8f vy = loadu(_y + i);
        const vec8f vz = avx2::add(vx, vy);
        storeu(_z + i, vz);
    }
    for (; i < idx; ++i) _z[i] = _x[i] + _y[i];
}

void matrix_t::sub(const f32 *_x, const f32 *_y, f32 *_z, const size_t idx) {
    size_t i = 0;

    for (; i + 8 < idx; i += 8) {
        const vec8f vx = loadu(_x + i);
        const vec8f vy = loadu(_y + i);
        const vec8f vz = avx2::sub(vx, vy);
        storeu(_z + i, vz);
    }
    for (; i < idx; ++i) _z[i] = _x[i] - _y[i];
}

void matrix_t::mul(const f32 *_x, const f32 *_y, f32 *_z, const size_t idx) {
    size_t i = 0;

    for (; i + 8 < idx; i += 8) {
        const vec8f vx = loadu(_x + i);
        const vec8f vy = loadu(_y + i);
        const vec8f vz = avx2::mul(vx, vy);
        storeu(_z + i, vz);
    }
    for (; i < idx; ++i) _z[i] = _x[i] * _y[i];
}

void matrix_t::div(const f32 *_x, const f32 *_y, f32 *_z, const size_t idx) {
    size_t i = 0;

    for (; i + 8 < idx; i += 8) {
        const vec8f vx = loadu(_x + i);
        const vec8f vy = loadu(_y + i);
        const vec8f vz = avx2::div(vx, vy);
        storeu(_z + i, vz);
    }
    for (; i < idx; ++i) _z[i] = _x[i] / _y[i];
}

void matrix_t::fma(const f32 *_x, const f32 *_y, const f32 *_z, f32 *_m, const size_t idx) {
    size_t i = 0;

    for (; i + 8 < idx; i += 8) {
        const vec8f vx = loadu(_x + i);
        const vec8f vy = loadu(_y + i);
        const vec8f vz = loadu(_z + i);
        const vec8f vm = avx2::fma(vx, vy, vz);
        storeu(_m + i, vm);
    }
    for (; i < idx; ++i) _m[i] = _x[i] * _y[i] + _z[i];
}

void matrix_t::matmul(const f32 *_x, const f32 *_y, f32 *_z, const size_t xIdx, const size_t yIdx, const size_t zIdx) {
    constexpr size_t _size = 8;

    for (size_t i = 0; i < xIdx; i += _size) {
        const size_t im = std::min(xIdx - i, _size);
        for (size_t j = 0; j < yIdx; j += _size) {
            const size_t jm = std::min(yIdx - j, _size);

            vec8f va[_size];

            for (size_t k = 0; k < im; ++k) va[k] = zero();

            for (size_t k = 0; k < jm; k += _size) {
                vec8f vb;

                if (jm == _size) vb = loadu(_y + k * zIdx + j);
                else vb = partial::load(_y + k * yIdx + j, jm);

                for (size_t l = 0; l < im; ++l) {
                    const vec8f vc = set(_x[(i + l) * zIdx + k]);
                    va[l] = avx2::fma(vc, vb, va[l]);
                }
            }

            for (size_t k = 0; k < im; ++k) {
                if (jm == _size) storeu(_z + (i + k) * zIdx + j, va[k]);
                else partial::store(_z + (i + k) * zIdx + j, va[k], jm);
            }
        }
    }
}
