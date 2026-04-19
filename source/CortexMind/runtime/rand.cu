//
// Created by muham on 19.04.2026.
//

#include "CortexMind/runtime/rand.cuh"
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::runtime;
using namespace cortex::_fw;

CurandContext& CurandContext::instance() {
    static CurandContext ctx;
    return ctx;
}

void CurandContext::init(const u64 seed) {
    CXM_ASSERT(
        curandCreateGenerator(&this->generator, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
        "CurandContext::init()",
        "curandCreateGenerator() failed"
    );
    CXM_ASSERT(
        curandSetPseudoRandomGeneratorSeed(this->generator, seed) == CURAND_STATUS_SUCCESS,
        "CurandContext::init()",
        "curandSetPseudoRandomGeneratorSeed() failed"
    );
}

void CurandContext::destroy() const {
    CXM_ASSERT(
        curandDestroyGenerator(this->generator) == CURAND_STATUS_SUCCESS,
        "CurandContext::destroy()",
        "curandDestroyGenerator() failed"
    );
}