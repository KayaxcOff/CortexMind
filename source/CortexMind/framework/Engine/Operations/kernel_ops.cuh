//
// Created by muham on 11.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH

#include <CortexMind/framework/Engine/Operations/base.cuh>
#include <CortexMind/runtime/macros.hpp>
#include <tlx/types.hpp>

namespace cortex::_fw::ops {
    struct Addition : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Subtraction : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Multiplication : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Division : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct  Square : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Power : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Sqrt : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct RSqrt : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Log : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Exp : KernelBase {
        [[nodiscard]]
        CXM_DEVICE_ATTR float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE_ATTR tlx::half operator()(tlx::half Xx) const noexcept;
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH