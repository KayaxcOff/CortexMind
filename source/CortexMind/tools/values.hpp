//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_TOOLS_VALUES_HPP
#define CORTEXMIND_TOOLS_VALUES_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <CortexMind/tools/types.hpp>

namespace cortex {
    inline auto host = _fw::sys::DeviceType::kHOST;
    inline auto cuda = _fw::sys::DeviceType::kCUDA;

    inline int32 exit       = CXM_EXIT;
    inline int32 err_exit   = CXM_ERR_EXIT;
} //namespace cortex

#endif //CORTEXMIND_TOOLS_VALUES_HPP