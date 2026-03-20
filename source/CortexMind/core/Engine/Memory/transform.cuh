//
// Created by muham on 14.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_MEMORY_TRANSFORM_CUH
#define CORTEXMIND_CORE_ENGINE_MEMORY_TRANSFORM_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Tools/err.hpp>

namespace cortex::_fw::sys {
    /**
     * @brief   Templated host-to-device and device-to-host memory copy utilities
     *
     * @tparam  T   Data type being transferred (float, int, half, etc.)
     */
    template<typename T>
    struct transform {
        /**
         * @brief   Uploads data from host memory to device memory
         * @param   host   Source pointer in host memory
         * @param   cuda   Destination pointer in device memory
         * @param   size   Number of **bytes** to copy (not number of elements!)
         *
         * @pre     host and cuda are valid pointers
         * @pre     size is multiple of sizeof(T) (caller responsibility)
         * @pre     cuda pointer allocated with cudaMalloc
         * @note    Synchronous operation – blocks until transfer complete
         * @note    Errors checked via CXM_CUDA_ASSERT
         */
        static void upload(const T* host, T* cuda, size_t size) {
            auto result = cudaMemcpy(cuda, host, size * sizeof(T), cudaMemcpyHostToDevice);
            CXM_ASSERT(result == cudaSuccess, "cortex::_fw::sys::transform::upload()", "cudaMemcpy failed");
        }
        /**
         * @brief   Downloads data from device memory to host memory
         * @param   host   Destination pointer in host memory
         * @param   cuda   Source pointer in device memory
         * @param   size   Number of **bytes** to copy
         *
         * @pre     cuda pointer allocated and filled on device
         * @pre     host has enough allocated space
         * @note    Synchronous – blocks until transfer finishes
         */
        static void download(T* host, const T* cuda, size_t size) {
            auto result = cudaMemcpy(host, cuda, size * sizeof(T), cudaMemcpyDeviceToHost);
            CXM_ASSERT(result == cudaSuccess, "cortex::_fw::sys::transform::download()", "cudaMemcpy failed");
        }
        /**
         * @brief   Copies data between two host pointers (std::memcpy)
         * @param   src    Source host pointer
         * @param   dst    Destination host pointer
         * @param   size   Number of **elements** to copy
         *
         * @note    Simple wrapper around std::memcpy
         * @note    No CUDA involvement — pure host operation
         */
        static void copy_h2h(const T* src, T* dst, size_t size) {
            std::memcpy(dst, src, size * sizeof(T));
        }
        /**
         * @brief   Copies data between two device pointers (cudaMemcpyDeviceToDevice)
         * @param   src    Source device pointer
         * @param   dst    Destination device pointer
         * @param   size   Number of **elements** to copy
         *
         * @pre     src and dst are valid device pointers
         * @note    Synchronous device-to-device copy
         * @note    Useful for internal buffer copies
         * @note    Throws via CXM_ASSERT on failure
         */
        static void copy_d2d(const T* src, T* dst, size_t size) {
            cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
        }
    };
} // namespace cortex::_fw::sys

#endif //CORTEXMIND_CORE_ENGINE_MEMORY_TRANSFORM_CUH