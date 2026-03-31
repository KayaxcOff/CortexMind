//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/AVX2/partial.hpp"
#include <CortexMind/core/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

vec8f partial::load(const f32 *dest, const size_t N) {
    vec8f output = zero();
    for (size_t i = 0; i < N; ++i) {
        reinterpret_cast<f32*>(&output)[i] = dest[i];
    }
    return output;
}

void partial::store(f32 *dest, vec8f val, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        dest[i] = reinterpret_cast<f32*>(&val)[i];
    }
}
