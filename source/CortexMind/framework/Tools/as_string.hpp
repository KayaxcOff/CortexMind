//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_AS_STRING_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_AS_STRING_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/framework/Tools/broadcast_kind.hpp>
#include <string>

namespace cortex::_fw {
    /**
     * @brief Converts a `DeviceType` enum to its string representation.
     */
    [[nodiscard]]
    std::string as_string(sys::DeviceType d_type);
    /**
     * @brief Converts a `BroadcastKind` enum to its string representation.
     */
    [[nodiscard]]
    std::string as_string(BroadcastKind kind);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_AS_STRING_HPP