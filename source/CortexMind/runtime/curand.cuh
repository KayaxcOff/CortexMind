//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_RUNTIME_RAND_CUH
#define CORTEXMIND_RUNTIME_RAND_CUH

#include <curand.h>

namespace cortex::_fw::runtime {
    struct Curand {
        explicit Curand(unsigned long long seed);
        ~Curand();

        curandGenerator_t rand_handle;

        static Curand& instance();

        static void reseed(unsigned long long seed)
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_RAND_CUH