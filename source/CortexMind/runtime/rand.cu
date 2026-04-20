//
// Created by muham on 19.04.2026.
//

#include "CortexMind/runtime/rand.cuh"
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::runtime;
using namespace cortex;

RandEngine& RandEngine::instance() {
    static RandEngine rand;
    return rand;
}

RandEngine::RandEngine(uint64 seed) {
    CXM_ASSERT(
        curandCreateGenerator(&this->generator, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
        "cortex::_fw::runtime::RandEngine::RandEngine()",
        "curandCreateGenerator() failed"
    );
    CXM_ASSERT(
        curandSetPseudoRandomGeneratorSeed(this->generator, seed) == CURAND_STATUS_SUCCESS,
        "cortex::_fw::runtime::RandEngine::RandEngine()",
        "curandSetPseudoRandomGeneratorSeed() failed"
    );
}

RandEngine::~RandEngine() {
    CXM_ASSERT(
        curandDestroyGenerator(this->generator) == CURAND_STATUS_SUCCESS,
        "cortex::_fw::runtime::RandEngine::~RandEngine()",
        "curandDestroyGenerator() failed"
    );
}