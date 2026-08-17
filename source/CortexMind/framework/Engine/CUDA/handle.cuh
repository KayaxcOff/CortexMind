//
// Created by muham on 17.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_HANDLE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_HANDLE_CUH

#include <cublas_v2.h>

namespace cortex::_fw::nv {
    /**
     * @brief RAII wrapper for a cuBLAS library handle.
     *
     * Owns a @c cublasHandle_t and manages its lifetime through construction
     * and destruction. The handle is created when the wrapper is constructed
     * and destroyed automatically when the wrapper goes out of scope.
     *
     * The wrapper is non-copyable and movable, allowing ownership of the
     * underlying cuBLAS handle to be transferred without duplicating it.
     */
    struct handle {
        /**
         * @brief Creates and initializes a cuBLAS handle.
         *
         * The underlying handle is created through @c cublasCreate.
         *
         * Terminates the application if cuBLAS handle creation fails.
         */
        handle();
        handle(const handle&) = delete;
        handle(handle&& other) noexcept;
        ~handle();

        /**
         * @brief Returns the underlying cuBLAS handle.
         *
         * Allows the wrapper to be passed directly to cuBLAS API functions
         * expecting a @c cublasHandle_t.
         *
         * @return The underlying cuBLAS handle.
         */
        [[nodiscard]]
        operator cublasHandle_t() const noexcept;

        handle& operator=(const handle&) = delete;
        handle& operator=(handle&& other) noexcept;
    private:
        cublasHandle_t m_value;
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_HANDLE_CUH