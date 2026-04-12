//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <string_view>

namespace cortex::_fw::sys {
    /**
     * @brief Specifies driver information as a string.
     * @param d_type Device as enum
     * @return Device as string
     */
    [[nodiscard]]
    std::string_view DeviceAsString(deviceType d_type);
} //namespace cortex::_fw::sys

#endif //CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP