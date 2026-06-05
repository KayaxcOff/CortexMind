//
// Created by muham on 16.05.2026.
//

#include "CortexMind/runtime/curand.cuh"
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::runtime;

Curand::Curand(unsigned long long seed) {
    CXM_ASSERT(
        curandCreateGenerator(&this->rand_handle, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS,
        "curandCreateGenerator() failed"
    );

    CXM_ASSERT(
        curandSetPseudoRandomGeneratorSeed(instance().rand_handle, seed) != CURAND_STATUS_SUCCESS,
        "curandSetPseudoRandomGeneratorSeed() failed"
    );
}

Curand::~Curand() {
    CXM_ASSERT(
        curandDestroyGenerator(this->rand_handle) != CURAND_STATUS_SUCCESS,
        "curandDestroyGenerator() failed"
    );
}

Curand& Curand::instance() {
    static Curand curand{42ULL};
    return curand;
}