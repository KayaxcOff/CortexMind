//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>

namespace cortex::_fw {
    struct BroadcastInfo {
        i32    ndim;
        size_t shape[CXM_MAX_DIMS];
        size_t stride_x[CXM_MAX_DIMS];
        size_t stride_y[CXM_MAX_DIMS];
        size_t stride_z[CXM_MAX_DIMS];
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP