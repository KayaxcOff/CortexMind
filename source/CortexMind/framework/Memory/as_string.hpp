//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP

#include <CortexMind/framework/Memory/type.hpp>
#include <tlx/string.hpp>

namespace cortex::_fw {
    /**
     * @brief Converts a device type to its canonical string representation.
     *
     * Returns the human-readable name associated with the specified
     * @ref DeviceType. The returned string is intended for logging,
     * debugging, serialization, and diagnostic output.
     *
     * If the specified device type is unknown or unsupported,
     * the function returns `"unknown"`.
     *
     * @param type The device type to convert.
     *
     * @return A string containing the canonical device name.
     *
     * @note The returned names follow the CortexMind runtime naming
     * convention (e.g. `"cpu"` and `"cuda"`).
     */
    [[nodiscard]]
    tlx::vstring as_string(DeviceType type);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP