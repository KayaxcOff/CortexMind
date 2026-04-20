//
// Created by muham on 20.04.2026.
//

#ifndef CORTEXMIND_RUNTIME_PROVIDER_CUH
#define CORTEXMIND_RUNTIME_PROVIDER_CUH

#include <CortexMind/tools/params.hpp>
#include <cublas_v2.h>

namespace cortex::_fw::runtime {
    /**
     * @brief Singleton provider for cuBLAS handle and CUDA context management.
     *
     * This class manages the lifetime of a cuBLAS handle and ensures the correct
     * CUDA device is selected. It follows the singleton pattern to provide
     * global access to the cuBLAS context throughout the application.
     */
    struct Provider {
        cublasHandle_t handle;

        /**
         * @brief Returns the singleton instance of the Provider.
         * @return Reference to the static Provider instance
         */
        static Provider& instance();
    private:
        Provider(int32 device_id = 0);
        ~Provider();
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_PROVIDER_CUH