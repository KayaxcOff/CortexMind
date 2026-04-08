//
// Created by muham on 7.04.2026.
//

#include "CortexMind/core/Engine/AVX2/partial.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

vec8f partial::load(const f32 *src, const size_t N) {
    vec8f output = zero();
    for (size_t i = 0; i < N; ++i) {
        reinterpret_cast<f32*>(&output)[i] = src[i];
    }
    return output;
}

void partial::store(f32 *dst, const vec8f src, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        dst[i] = reinterpret_cast<const f32*>(&src)[i];
    }
}