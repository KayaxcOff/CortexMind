//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP

#include <CortexMind/framework/Memory/type.hpp>
#include <tlx/string.hpp>

namespace cortex::_fw {
    [[nodiscard]]
    tlx::vstring as_string(DeviceType type);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_DEVICE_AS_STRING_HPP