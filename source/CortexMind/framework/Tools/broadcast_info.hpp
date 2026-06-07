//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>

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
        i64 shape[CXM_MAX_DIMS];                       ///< C-style array: Broadcasted common shape
        i64 stride_x[CXM_MAX_DIMS];                    ///< Strides for first input
        i64 stride_y[CXM_MAX_DIMS];                    ///< Strides for second input
        i64 stride_z[CXM_MAX_DIMS];                    ///< Strides for output
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_INFO_HPP