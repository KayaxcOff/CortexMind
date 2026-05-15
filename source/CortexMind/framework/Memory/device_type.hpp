//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>

namespace cortex::_fw::sys {
    /**
     * @brief Specifies the memory location / device type.
     *
     * Used by memory managers, tensors, and buffers to determine
     * where the data physically resides.
     */
    enum class DeviceType : i64 {
        kHOST = CXM_HOST_DEVICE,   ///< Host (CPU) memory (RAM)
        kCUDA = CXM_CUDA_DEVICE    ///< CUDA device (GPU) memory (VRAM)
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP