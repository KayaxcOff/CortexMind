//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH
#define CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH

#include <cwchar>

namespace cortex::_fw {
    /**
     * @brief Collection of helper functions for memory transfers.
     *
     * Provides wrappers around host and CUDA memory copy operations.
     * The functions centralize data movement between host and device
     * memory while performing runtime CUDA error checking when required.
     */
    struct transform {
        /**
         * @brief Copies data from host memory to CUDA device memory.
         *
         * @param dst Destination device memory.
         * @param src Source host memory.
         * @param byte Number of bytes to copy.
         */
        static void upload(void* dst, const void* src, std::size_t byte);
        /**
         * @brief Copies data from CUDA device memory to host memory.
         *
         * @param dst Destination host memory.
         * @param src Source device memory.
         * @param byte Number of bytes to copy.
         */
        static void download(void* dst, const void* src, std::size_t byte);
        /**
         * @brief Copies data between two host memory regions.
         *
         * Performs a standard host-to-host memory copy using memcpy().
         *
         * @param dst Destination host memory.
         * @param src Source host memory.
         * @param byte Number of bytes to copy.
         */
        static void copy_h2h(void* dst, const void* src, std::size_t byte);
        /**
         * @brief Copies data between two CUDA device memory regions.
         *
         * @param dst Destination device memory.
         * @param src Source device memory.
         * @param byte Number of bytes to copy.
         */
        static void copy_d2d(void* dst, const void* src, std::size_t byte);
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH