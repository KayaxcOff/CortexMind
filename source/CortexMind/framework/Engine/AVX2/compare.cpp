//
// Created by muham on 10.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/compare.hpp"
#include <CortexMind/framework/Engine/AVX2/cmp.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

namespace {
    /**
     * @brief Converts an AVX2 floating-point comparison mask to integer values.
     *
     * AVX2 floating-point comparison instructions produce a mask in which
     * each lane is represented by either all zero bits or all one bits.
     * This function extracts the sign bit of each lane and converts the
     * comparison result into a `0` or `1` integer value.
     *
     * @param mask AVX2 comparison mask.
     * @return AVX2 integer vector containing `1` for true lanes and `0`
     *         for false lanes.
     */
    [[nodiscard]]
    vec8i to_i32(const vec8f& mask) {
        const vec8i bits = _mm256_castps_si256(mask);
        return _mm256_srli_epi32(bits, 31);
    }
} //unnamed namespace

void compare::gt(const float *Xx, const float *Xy, std::int32_t *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i + 1);
        const vec8f mask = cmp::gt(r0, r1);
        const vec8i output = to_i32(mask);
        storeu(Xz + i, output);
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] > Xy[i] ? 1 : 0;
    }
}

void compare::lt(const float *Xx, const float *Xy, std::int32_t *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f mask = cmp::lt(r0, r1);
        const vec8i output = to_i32(mask);
        storeu(Xz + i, output);
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] < Xy[i] ? 1 : 0;
    }
}

void compare::eq(const float *Xx, const float *Xy, std::int32_t *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f mask = cmp::eq(r0, r1);
        const vec8i output = to_i32(mask);
        storeu(Xz + i, output);
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] == Xy[i] ? 1 : 0;
    }
}

void compare::ge(const float *Xx, const float *Xy, std::int32_t *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f mask = cmp::ge(r0, r1);
        const vec8i output = to_i32(mask);
        storeu(Xz + i, output);
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] >= Xy[i] ? 1 : 0;
    }
}

void compare::le(const float *Xx, const float *Xy, std::int32_t *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f mask = cmp::le(r0, r1);
        const vec8i output = to_i32(mask);
        storeu(Xz + i, output);
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] <= Xy[i] ? 1 : 0;
    }
}

void compare::neq(const float *Xx, const float *Xy, std::int32_t *Xz, const std::size_t N) {
    std::size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f r0 = loadu(Xx + i);
        const vec8f r1 = loadu(Xy + i);
        const vec8f mask = cmp::neq(r0, r1);
        const vec8i output = to_i32(mask);
        storeu(Xz + i, output);
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] != Xy[i] ? 1 : 0;
    }
}