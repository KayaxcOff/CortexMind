//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP

#include <cstdint>

namespace cortex::_fw {
    /**
     * @brief Identifies the physical device on which tensor data is stored.
     *
     * DeviceType specifies the execution and memory location associated with
     * tensors and other runtime objects. It is used by the framework to select
     * the appropriate memory allocator, execution backend, and data transfer
     * operations.
     *
     * The device type is independent of the tensor's scalar data type
     * (see @ref DType).
     */
    enum class DeviceType : std::uint8_t {
        Unknown = 0,
        HOST = 1,
        CUDA = 2
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP