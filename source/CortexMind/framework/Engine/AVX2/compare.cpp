//
// Created by muham on 6.06.2026.
//

#include "CortexMind/framework/Engine/AVX2/compare.hpp"
#include <CortexMind/framework/Engine/AVX2/cmp.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

void compare::gt(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) noexcept {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f res = cmp::gt(r0, r1);
        const i32 mask = cmp::mask(res);
        for (size_t j = 0; j < 8; ++j) {
            Xz[i + j] = ((mask & (1 << j)) != 0) ? 1.0f : 0.0f;
        }
    }
    for (; i < N; ++i) {
        Xz[i] = (Xx[i] > Xy[i]) ? 1.0f : 0.0f;
    }
}

void compare::lt(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) noexcept {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f res = cmp::lt(r0, r1);
        const i32 mask = cmp::mask(res);
        for (size_t j = 0; j < 8; ++j) {
            Xz[i + j] = ((mask & (1 << j)) != 0) ? 1.0f : 0.0f;
        }
    }
    for (; i < N; ++i) {
        Xz[i] = (Xx[i] < Xy[i]) ? 1.0f : 0.0f;
    }
}

void compare::eq(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) noexcept {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f res = cmp::eq(r0, r1);
        const i32 mask = cmp::mask(res);
        for (size_t j = 0; j < 8; ++j) {
            Xz[i + j] = (mask & (1 << j)) != 0;
        }
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] == Xy[i];
    }
}

void compare::ge(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) noexcept {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f res = cmp::eq(r0, r1);
        const i32 mask = cmp::mask(res);
        for (size_t j = 0; j < 8; ++j) {
            Xz[i + j] = ((mask & (1 << j)) != 0) ? 1.0f : 0.0f;
        }
    }
    for (; i < N; ++i) {
        Xz[i] = (Xx[i] == Xy[i]) ? 1.0f : 0.0f;
    }
}

void compare::le(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) noexcept {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f res = cmp::ge(r0, r1);
        const i32 mask = cmp::mask(res);
        for (size_t j = 0; j < 8; ++j) {
            Xz[i + j] = ((mask & (1 << j)) != 0) ? 1.0f : 0.0f;
        }
    }
    for (; i < N; ++i) {
        Xz[i] = (Xx[i] >= Xy[i]) ? 1.0f : 0.0f;
    }
}

void compare::neq(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) noexcept {
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f res = cmp::neq(r0, r1);
        const i32 mask = cmp::mask(res);
        for (size_t j = 0; j < 8; ++j) {
            Xz[i + j] = ((mask & (1 << j)) != 0) ? 1.0f : 0.0f;
        }
    }
    for (; i < N; ++i) {
        Xz[i] = (Xx[i] != Xy[i]) ? 1.0f : 0.0f;
    }
}