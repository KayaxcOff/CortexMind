//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_FUNCTIONS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_FUNCTIONS_HPP

#include <CortexMind/core/Engine/AVX2/params.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    [[nodiscard]]
    vec8f load(const f32* src);
    void store(f32* dst, vec8f src);

    [[nodiscard]]
    vec8f loadu(const f32* src);
    void storeu(f32* dst, vec8f src);

    [[nodiscard]]
    vec8f set1(f32 value);
    [[nodiscard]]
    vec8f zero();

    [[nodiscard]]
    vec8f add(vec8f Xx, vec8f Xy);
    [[nodiscard]]
    vec8f sub(vec8f Xx, vec8f Xy);
    [[nodiscard]]
    vec8f mul(vec8f Xx, vec8f Xy);
    [[nodiscard]]
    vec8f div(vec8f Xx, vec8f Xy);

    [[nodiscard]]
    vec8f exp(vec8f Xx);
    [[nodiscard]]
    vec8f log(vec8f Xx);
    [[nodiscard]]
    vec8f sqrt(vec8f Xx);
    [[nodiscard]]
    vec8f abs(vec8f Xx);
    [[nodiscard]]
    vec8f neg(vec8f Xx);
    [[nodiscard]]
    vec8f tanh(vec8f Xx);
    [[nodiscard]]
    vec8f pow(vec8f Xx, vec8f Xy);
    [[nodiscard]]
    vec8f fma(vec8f Xx, vec8f Xy, vec8f Xz);
    [[nodiscard]]
    vec8f rcp_nr(vec8f Xx);

    [[nodiscard]]
    vec8f blendv(vec8f Xx, vec8f Xy, vec8f Xz);
    [[nodiscard]]
    vec8f min(vec8f Xx, vec8f Xy);
    [[nodiscard]]
    vec8f max(vec8f Xx, vec8f Xy);

    [[nodiscard]]
    f32 hsum(vec8f Xx);
    [[nodiscard]]
    f32 hmax(vec8f Xx);
    [[nodiscard]]
    f32 hmin(vec8f Xx);

    [[nodiscard]]
    vec8f relu(vec8f Xx);
    [[nodiscard]]
    vec8f leaky_relu(vec8f Xx, f32 alpha = 0.01f);
    [[nodiscard]]
    vec8f sigmoid(vec8f Xx);
    [[nodiscard]]
    vec8f sigmoid_fast(vec8f Xx);
    [[nodiscard]]
    vec8f gelu_exact(vec8f Xx);
    [[nodiscard]]
    vec8f gelu(vec8f Xx);
    [[nodiscard]]
    vec8f silu(vec8f Xx);
    [[nodiscard]]
    vec8f swish(vec8f Xx, f32 beta = 1.0f);
    [[nodiscard]]
    vec8f softmax(vec8f Xx);
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_FUNCTIONS_HPP