//
// Created by muham on 11.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH

#include <CortexMind/framework/Engine/Operations/base.cuh>
#include <tlx/types.hpp>

namespace cortex::_fw::ops {
    struct Addition : KernelBase {
        [[nodiscard]]
        float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH