//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_HPP

#include <CortexMind/core/Tools/utils.cuh>
#include <cstring>   // for std::memcpy

namespace cortex::_fw::sys {
    /**
     * @brief Memory transfer utilities between host and device.
     *
     * Provides convenient static functions for uploading data from host to device,
     * downloading from device to host, and performing copies within the same memory space.
     *
     * @tparam T Data type to transfer (usually float, int, etc.)
     */
    template<typename T>
    struct transform {
        /**
         * @brief Uploads data from host to device (Host → Device).
         * @param src   Source pointer on host
         * @param dst   Destination pointer on device
         * @param count Number of elements to transfer
         */
        static void upload(const T* src, T* dst, const size_t count) {
            cuda::memcpy(dst, src, sizeof(T) * count, cuda::CXM_HOST_TO_DEVICE);
        }
        /**
         * @brief Downloads data from device to host (Device → Host).
         * @param src   Source pointer on device
         * @param dst   Destination pointer on host
         * @param count Number of elements to transfer
         */
        static void download(const T* src, T* dst, const size_t count) {
            cuda::memcpy(dst, src, sizeof(T) * count, cuda::CXM_DEVICE_TO_HOST);
        }
        /**
         * @brief Copies data from device to device (Device → Device).
         * @param src   Source pointer on device
         * @param dst   Destination pointer on device
         * @param count Number of elements to copy
         */
        static void copy_d2d(const T* src, T* dst, const size_t count) {
            cuda::memcpy(dst, src, sizeof(T) * count, cuda::CXM_DEVICE_TO_DEVICE);
        }
        /**
         * @brief Copies data from host to host using std::memcpy (Host → Host).
         * @param src   Source pointer on host
         * @param dst   Destination pointer on host
         * @param count Number of elements to copy
         */
        static void copy_h2h(const T* src, T* dst, const size_t count) {
            std::memcpy(dst, src, sizeof(T) * count);
        }
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_HPP