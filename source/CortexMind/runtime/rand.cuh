//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_RUNTIME_RAND_CUH
#define CORTEXMIND_RUNTIME_RAND_CUH

#include <CortexMind/tools/params.hpp>
#include <curand.h>

namespace cortex::_fw::runtime {
    /**
     * @brief Singleton random number generator engine based on cuRAND.
     *
     * Provides a global, thread-safe cuRAND generator for random number generation
     * on the GPU. Uses the singleton pattern to ensure a single generator instance
     * throughout the application lifetime.
     */
    struct RandEngine {
        curandGenerator_t generator;

        /**
         * @brief Returns the singleton instance of the random engine.
         * @return Reference to the static RandEngine instance
         */
        static RandEngine& instance();
    private:
        RandEngine(uint64 seed = 42ULL);
        ~RandEngine();
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_RAND_CUH