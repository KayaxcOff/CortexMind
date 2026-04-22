//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <string>

namespace cortex::_fw {
    /**
     * @brief Specifies driver information as a string.
     * @param d_type Device as enum
     * @return Device as string
     */
    [[nodiscard]]
    const std::string& DeviceAsString(sys::deviceType d_type);
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP