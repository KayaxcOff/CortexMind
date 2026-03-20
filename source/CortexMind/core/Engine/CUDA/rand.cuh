//
// Created by muham on 17.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_RAND_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_RAND_CUH

#include <CortexMind/core/Tools/params.hpp>
#include <CortexMind/core/Tools/err.hpp>
#include <curand.h>
#include <random>

namespace cortex::_fw::cuda {
    /**
     * @brief   Random number generation utilities using cuRAND
     *
     * Manages a single cuRAND generator instance (singleton) and provides
     * convenient methods for common distributions.
     */
    struct rand {
        /**
         * @brief   Generates uniform random numbers in [min, max] (inclusive)
         * @param   Xx      Device output array (will be overwritten)
         * @param   min     Lower bound (inclusive)
         * @param   max     Upper bound (inclusive)
         * @param   idx     Number of elements to generate
         *
         * @pre     Xx is valid device pointer with space for ≥ idx floats
         * @pre     min ≤ max
         * @note    Uses curandGenerateUniform (uniform in [0,1)) then scales/shifts
         * @note    Scaling/shifting uses inplace_scalar::mul/add (efficient)
         * @note    Synchronous generation — blocks until complete
         * @note    For min=0, max=1 no scaling is applied (direct generation)
         */
        static void uniform(f32* __restrict__ Xx, f32 min, f32 max, size_t idx);

    private:
        /**
         * @brief   RAII guard for cuRAND generator lifetime
         *
         * Creates generator on construction, destroys on destruction.
         * Uses default pseudo-random RNG with host random seed.
         */
        struct RandGuard {
            curandGenerator_t gen;
            RandGuard() {
                CXM_ASSERT(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
                    "rand_t::RandGuard()", "Failed to create cuRAND generator");
                CXM_ASSERT(curandSetPseudoRandomGeneratorSeed(gen, std::random_device{}()) == CURAND_STATUS_SUCCESS,
                    "rand_t::RandGuard()", "Failed to set seed");
            }
            ~RandGuard() { curandDestroyGenerator(gen); }
        };
        /**
         * @brief   Returns the singleton cuRAND generator
         * @return  curandGenerator_t (lazy-initialized)
         *
         * @note    Generator is created once on first call (static local)
         * @note    Thread-safe (cuRAND generator is thread-safe)
         */
        static curandGenerator_t get_generator();
    };
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_RAND_CUH