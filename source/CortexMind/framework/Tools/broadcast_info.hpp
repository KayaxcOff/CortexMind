//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <array>

namespace cortex::_fw {
    /**
     * @brief Contains precomputed broadcasting information for two input tensors.
     *
     * This struct is typically filled by `classify_broadcast()` and used by
     * dispatchers and kernels to efficiently handle broadcasting without
     * dynamic shape checks inside hot loops.
     */
    struct BroadcastInfo {
        i32 ndim;                                      ///< Number of dimensions after broadcasting
        std::array<i64, CXM_MAX_DIMS> shape;           ///< Broadcasted common shape
        std::array<i64, CXM_MAX_DIMS> stride_x;        ///< Strides for first input
        std::array<i64, CXM_MAX_DIMS> stride_y;        ///< Strides for second input
        std::array<i64, CXM_MAX_DIMS> stride_z;        ///< Strides for output
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP