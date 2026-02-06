//
// Created by muham on 2.02.2026.
//

#include "CortexMind/core/Engine/AVX/partial.hpp"
#include <CortexMind/core/Engine/AVX/funcs.hpp>

using namespace cortex::_fw::avx2;

vec8f partial::load(const f32 *dest, const size_t idx) {
    vec8f output = zero();

    for (size_t i = 0; i < idx; ++i) reinterpret_cast<f32*>(&output)[i] = dest[i];

    return output;
}

void partial::store(f32 *dest, vec8f val, const size_t idx) {
    for (size_t i = 0; i < idx; ++i) dest[i] = reinterpret_cast<f32*>(&val)[i];
}
