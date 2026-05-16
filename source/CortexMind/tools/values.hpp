//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_TOOLS_VALUES_HPP
#define CORTEXMIND_TOOLS_VALUES_HPP

#include <CortexMind/framework/Memory/device_type.hpp>

namespace cortex {
    inline auto host = _fw::sys::DeviceType::kHOST;
    inline auto cuda = _fw::sys::DeviceType::kCUDA;
} //namespace cortex

#endif //CORTEXMIND_TOOLS_VALUES_HPP