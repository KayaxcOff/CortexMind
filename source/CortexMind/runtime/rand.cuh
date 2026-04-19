//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_RUNTIME_RAND_CUH
#define CORTEXMIND_RUNTIME_RAND_CUH

#include <CortexMind/framework/Tools/params.hpp>
#include <curand.h>

namespace cortex::_fw::runtime {
    /**
     * @brief Singleton context manager for cuRAND.
     *
     * Provides a thread-safe singleton instance to manage the cuRAND generator.
     * Handles initialization, destruction and seed management.
     */
    struct CurandContext {
        curandGenerator_t generator;

        /**
         * @brief Returns the singleton instance of CurandContext.
         */
        static CurandContext& instance();

        /**
         * @brief Initializes the cuRAND generator with the given seed.
         * @param seed Random seed (default: 42)
         */
        void init(_fw::u64 seed = 42ULL);

        /**
         * @brief Destroys the cuRAND generator.
         */
        void destroy() const;

        CurandContext(const CurandContext&)            = delete;
        CurandContext& operator=(const CurandContext&) = delete;
    private:
        CurandContext()  = default;
        ~CurandContext() = default;
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_RAND_CUH