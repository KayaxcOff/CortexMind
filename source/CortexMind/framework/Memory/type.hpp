//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP

#include <cstdint>

namespace cortex::_fw {
    enum class DeviceType : std::uint8_t {
        HOST,
        CUDA
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_TYPE_HPP