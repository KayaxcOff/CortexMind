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
        CXM_DEVICE float operator()(const float Xx, const float Xy) const noexcept {
            return Xx + Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
            return Xx + Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
            return Xx + Xy;
        }
    };

    struct Subtraction : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx, const float Xy) const noexcept {
            return Xx - Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
            return Xx - Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
            return Xx - Xy;
        }
    };

    struct Multiplication : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx, const float Xy) const noexcept {
            return Xx * Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
            return Xx * Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
            return Xx * Xy;
        }
    };

    struct Division : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx, const float Xy) const noexcept {
            return Xx / Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
            return Xx / Xy;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
            return Xx / Xy;
        }
    };

    struct Square : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return Xx * Xx;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return Xx * Xx;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return Xx * Xx;
        }
    };

    struct Power : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx, const float Xy) const noexcept {
            return powf(Xx, Xy);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
            return powf(Xx, Xy);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
            return powf(Xx, Xy);
        }
    };

    struct Sqrt : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return sqrtf(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return hsqrt(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return hsqrt(Xx);
        }
    };

    struct RSqrt : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return rsqrtf(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return hrsqrt(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return hrsqrt(Xx);
        }
    };

    struct Log : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return logf(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return hlog(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return hlog(Xx);
        }
    };

    struct Exp : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return expf(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return hexp(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return hexp(Xx);
        }
    };

    struct Erf : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return erff(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return erff(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return erff(Xx);
        }
    };

    struct Sin : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return sinf(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return hsin(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return hsin(Xx);
        }
    };

    struct Cos : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return cosf(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return hcos(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return hcos(Xx);
        }
    };

    struct Abs : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return fabsf(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return __habs(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return __habs(Xx);
        }
    };

    struct Neg : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return -Xx;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return __hneg(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return __hneg(Xx);
        }
    };

    struct Rcp : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return __frcp_rn(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return hrcp(Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return hrcp(Xx);
        }
    };

    struct Inverse : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return 1.f / Xx;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return tlx::bfloat16(1.f) / Xx;
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return tlx::half(1.f) / Xx;
        }
    };

    struct Sign : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx) const noexcept {
            return static_cast<float>((Xx > 0.f) - (Xx < 0.f));
        }

        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx) const noexcept {
            return tlx::bfloat16((Xx > tlx::bfloat16(0.f)) - (Xx < tlx::bfloat16(0.f)));
        }

        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx) const noexcept {
            return tlx::half((Xx > tlx::half(0.f)) - (Xx < tlx::half(0.f)));
        }
    };

    struct Lerp : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx, const float Xy, const float Xz) const noexcept {
            return fmaf(Xz, Xy - Xx, Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy, const tlx::bfloat16 Xz) const noexcept {
            return __hfma(Xz, Xy - Xx, Xx);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy, const tlx::half Xz) const noexcept {
            return __hfma(Xz, Xy - Xx, Xx);
        }
    };

    struct Max : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx, const float Xy) const noexcept {
            return fmaxf(Xx, Xy);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
            return __hmax(Xx, Xy);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
            return __hmax(Xx, Xy);
        }
    };

    struct Min : KernelBase {
        [[nodiscard]]
        CXM_DEVICE float operator()(const float Xx, const float Xy) const noexcept {
            return fminf(Xx, Xy);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::bfloat16 operator()(const tlx::bfloat16 Xx, const tlx::bfloat16 Xy) const noexcept {
            return __hmin(Xx, Xy);
        }
        [[nodiscard]]
        CXM_DEVICE tlx::half operator()(const tlx::half Xx, const tlx::half Xy) const noexcept {
            return __hmin(Xx, Xy);
        }
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_CUH