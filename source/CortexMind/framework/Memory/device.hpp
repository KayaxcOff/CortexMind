//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_HPP

#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/runtime/macros.hpp>

namespace cortex::_fw::sys {
    /**
     * @brief Specifies the driver.
     * @warning It can point to either
     * the CPU or the GPU.
     */
    enum deviceType : i64 {
        host = CXM_HOST_DEVICE,
        cuda = CXM_CUDA_DEVICE,
    };
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_HPP