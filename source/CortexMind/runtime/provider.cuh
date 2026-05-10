//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_RUNTIME_PROVIDER_CUH
#define CORTEXMIND_RUNTIME_PROVIDER_CUH

#include <CortexMind/framework/Tools/types.hpp>
#include <cublas_v2.h>

namespace cortex::_fw::runtime {
    /**
     * @brief Singleton provider for CUDA runtime resources.
     *
     * Manages cuBLAS handle and CUDA device context. This is the central point
     * for all cuBLAS operations in the framework.
     *
     * @note Thread-safe singleton (Meyers' Singleton).
     * @note Automatically initializes and cleans up cuBLAS handle.
     */
    struct Provider {
        cublasHandle_t handle; ///< cuBLAS handle

        /**
         * @brief Returns the global instance of the Provider.
         *
         * @return Reference to the singleton Provider instance.
         */
        static Provider& instance();
    private:
        /**
         * @brief Private constructor.
         *
         * @param device_id CUDA device ID to use (default = 0).
         */
        Provider(i32 device_id = 0);
        /**
         * @brief Destructor that cleans up cuBLAS resources.
         */
        ~Provider();
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_PROVIDER_CUH