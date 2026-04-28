//
// Created by muham on 28.04.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_BROADCAST_HPP
#define CORTEXMIND_CORE_TOOLS_BROADCAST_HPP

#include <CortexMind/runtime/macros.hpp>

namespace cortex::_fw {
    struct BroadcastInfo {
        int ndim;
        size_t shape[CXM_MAX_DIMS];
        size_t stride_x[CXM_MAX_DIMS];
        size_t stride_y[CXM_MAX_DIMS];
        size_t stride_z[CXM_MAX_DIMS];
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_BROADCAST_HPP