//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_RUNTIME_RAND_CUH
#define CORTEXMIND_RUNTIME_RAND_CUH

#include <CortexMind/tools/params.hpp>
#include <curand.h>

namespace cortex::_fw::runtime {
    struct RandEngine {
        curandGenerator_t generator;

        static RandEngine& instance();
    private:
        RandEngine(uint64 seed = 42ULL);
        ~RandEngine();
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_RAND_CUH