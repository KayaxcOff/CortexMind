//
// Created by muham on 11.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_BASE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_BASE_CUH

#include <CortexMind/runtime/macros.hpp>

namespace cortex::_fw::ops {
    struct KernelBase {
        CXM_DEVICE virtual ~KernelBase() = default;
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_BASE_CUH