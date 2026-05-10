//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH
#define CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::sys {
    /**
     * @brief Memory transfer and copy utilities.
     *
     * This struct offers convenient, type-safe wrappers for moving data between
     * host and device memory spaces, as well as within the same memory space.
     */
    struct transform {
        /**
         * @brief Uploads data from host to device (Host → Device).
         *
         * @param dst   Destination pointer (device memory)
         * @param src   Source pointer (host memory)
         * @param count Number of `f32` elements to copy
         */
        static void upload(f32* dst, const f32* src, size_t count);
        /**
         * @brief Downloads data from device to host (Device → Host).
         *
         * @param dst   Destination pointer (host memory)
         * @param src   Source pointer (device memory)
         * @param count Number of `f32` elements to copy
         */
        static void download(f32* dst, const f32* src, size_t count);
        /**
         * @brief Copies data between two host memory regions (Host → Host).
         *
         * Uses `std::memcpy` internally.
         *
         * @param dst   Destination pointer (host)
         * @param src   Source pointer (host)
         * @param count Number of `f32` elements to copy
         */
        static void copy_h2h(f32* dst, const f32* src, size_t count);
        /**
         * @brief Copies data between two device memory regions (Device → Device).
         *
         * Uses `cudaMemcpy` with `cudaMemcpyDeviceToDevice`.
         *
         * @param dst   Destination pointer (device)
         * @param src   Source pointer (device)
         * @param count Number of `f32` elements to copy
         */
        static void copy_d2d(f32* dst, const f32* src, size_t count);
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_TRANSFORM_CUH