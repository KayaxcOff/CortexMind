//
// Created by muham on 17.03.2026.
//

#include "CortexMind/core/Engine/CUDA/rand.cuh"
#include <CortexMind/core/Engine/CUDA/inplace.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

curandGenerator_t rand::get_generator() {
    static RandGuard guard;
    return guard.gen;
}

void rand::uniform(f32* __restrict__ Xx, const f32 min, const f32 max, const size_t idx) {
    if (idx == 0) return;
    CXM_ASSERT(min <= max, "rand_t::uniform()", "min must be <= max");

    curandGenerator_t gen = get_generator();
    CXM_ASSERT(curandGenerateUniform(gen, Xx, idx) == CURAND_STATUS_SUCCESS,
        "rand_t::uniform()", "Failed to generate uniform random numbers");

    if (min != 0.0f || max != 1.0f) {
        inplace_scalar::mul(Xx, max - min, idx);
        inplace_scalar::add(Xx, min, idx);
    }
}