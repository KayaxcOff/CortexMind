//
// Created by muham on 8.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP

#include <CortexMind/runtime/macros.hpp>
#include <tlx/vector.hpp>
#include <cstdint>

namespace cortex::_fw {
    struct BroadcastInfo {
        std::int32_t ndim;
        tlx::vec<std::int64_t, CXM_MAX_DIMS> shape;
        tlx::vec<std::int64_t, CXM_MAX_DIMS> stride_x;
        tlx::vec<std::int64_t, CXM_MAX_DIMS> stride_y;
        tlx::vec<std::int64_t, CXM_MAX_DIMS> stride_z;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP