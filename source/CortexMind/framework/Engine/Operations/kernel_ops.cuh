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
        CXM_DEVICE float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Subtraction : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Multiplication : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Division : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct  Square : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Power : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx, float Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx, tlx::half Xy) const noexcept;
    };

    struct Sqrt : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct RSqrt : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Log : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Exp : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Erf : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Sin : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Cos : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Abs : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Neg : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Rcp : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx) const noexcept;
    };

    struct Lerp : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(float Xx, float Xy, float Xz) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(tlx::bfloat16 Xx, tlx::bfloat16 Xy, tlx::bfloat16 Xz) const noexcept;
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(tlx::half Xx, tlx::half Xy, tlx::half Xz) const noexcept;
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH