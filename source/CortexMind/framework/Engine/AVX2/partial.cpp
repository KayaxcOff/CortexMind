//
// Created by muham on 8.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/partial.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>

using namespace cortex::_fw::avx2;

vec8f partial::load(const float *src, const std::size_t N) {
    vec8f output = zerof();
    for (std::size_t i = 0; i < N; ++i) {
        reinterpret_cast<float*>(&output)[i] = src[i];
    }
    return output;
}

void partial::store(float *dst, const vec8f &src, const std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        dst[i] = reinterpret_cast<const float*>(&src)[i];
    }
}