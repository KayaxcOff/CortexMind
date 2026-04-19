//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_TOOLS_DEVICE_HPP
#define CORTEXMIND_TOOLS_DEVICE_HPP

#include <CortexMind/framework/Memory/device.hpp>

namespace cortex {
    inline auto host = _fw::sys::deviceType::host;
    inline auto cuda = _fw::sys::deviceType::cuda;
} //namespace cortex

#endif //CORTEXMIND_TOOLS_DEVICE_HPP