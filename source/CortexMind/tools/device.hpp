//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_TOOLS_DEVICE_HPP
#define CORTEXMIND_TOOLS_DEVICE_HPP

#include <CortexMind/framework/Memory/device.hpp>

namespace cortex {
    /**
     * @brief It manages devices.
     * It has two variables: `cuda` and `host`.
     */
    using device_t = _fw::sys::deviceType;
} //namespace cortex

#endif //CORTEXMIND_TOOLS_DEVICE_HPP