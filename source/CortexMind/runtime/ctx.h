//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_RUNTIME_CTX_H
#define CORTEXMIND_RUNTIME_CTX_H

#include <CortexMind/framework/Tools/params.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <cublas_v2.h>
#endif //#if CXM_IS_CUDA_AVAILABLE

namespace cortex::_fw::runtime {
    /**
     * @brief Singleton context manager for cuBLAS.
     *
     * Provides a thread-safe singleton instance to manage the cuBLAS handle.
     * Handles initialization, destruction, and device selection for cuBLAS operations.
     */
    struct CublasContext {
        cublasHandle_t handle;

        /**
         * @brief Returns the singleton instance of CublasContext.
         * @return Reference to the static CublasContext instance
         */
        static CublasContext& instance();

        /**
         * @brief Initializes the cuBLAS context on the specified device.
         * @param device_id CUDA device ID (default: 0)
         */
        void init(i32 device_id = 0);
        /**
         * @brief Destroys the cuBLAS handle.
         *
         * Should be called before program termination to release resources.
         */
        void destroy() const;

        CublasContext(const CublasContext&) = delete;
        CublasContext& operator=(const CublasContext&) = delete;
    private:
        CublasContext() = default;
        ~CublasContext() = default;
    };
} //namespace cortex::_fw::runtime

#endif //CORTEXMIND_RUNTIME_CTX_H