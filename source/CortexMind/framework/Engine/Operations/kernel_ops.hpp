//
// Created by muham on 19.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_HPP

#include <CortexMind/framework/Engine/Operations/base.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <tlx/math.hpp>
#include <cmath>

namespace cortex::_fw::ops {
    struct Add : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx, const avx2::vec8f& Xy) const noexcept {
            return avx2::add(Xx, Xy);
        }
        [[nodiscard]]
        float operator()(const float Xx, const float Xy) const noexcept {
            return Xx + Xy;
        }
    };

    struct Subtract : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx, const avx2::vec8f& Xy) const noexcept {
            return avx2::sub(Xx, Xy);
        }
        [[nodiscard]]
        float operator()(const float Xx, const float Xy) const noexcept {
            return Xx - Xy;
        }
    };

    struct Multiply : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx, const avx2::vec8f& Xy) const noexcept {
            return avx2::mul(Xx, Xy);
        }
        [[nodiscard]]
        float operator()(const float Xx, const float Xy) const noexcept {
            return Xx * Xy;
        }
    };

    struct Divide : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx, const avx2::vec8f& Xy) const noexcept {
            return avx2::div(Xx, Xy);
        }
        [[nodiscard]]
        float operator()(const float Xx, const float Xy) const noexcept {
            return Xx / Xy;
        }
    };

    struct Square : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::square(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return Xx * Xx;
        }
    };

    struct Power : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx, const avx2::vec8f& Xy) const noexcept {
            return avx2::pow(Xx, Xy);
        }
        [[nodiscard]]
        float operator()(const float Xx, const float Xy) const noexcept {
            return std::pow(Xx, Xy);
        }
    };

    struct Sqrt : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::sqrt(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return std::sqrt(Xx);
        }
    };

    struct RSqrt : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::rsqrt(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return 1.f / std::sqrt(Xx);
        }
    };

    struct Log : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::log(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return std::log(Xx);
        }
    };

    struct Exp : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::exp(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return std::exp(Xx);
        }
    };

    struct Erf : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::erf(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return std::erf(Xx);
        }
    };

    struct Sin : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::sin(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return std::sin(Xx);
        }
    };

    struct Cos : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::cos(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return std::cos(Xx);
        }
    };

    struct Abs : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::abs(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return std::abs(Xx);
        }
    };

    struct Neg : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::neg(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return -Xx;
        }
    };

    struct Rcp : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::rcp(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return 1.f / Xx;
        }
    };

    struct Sign : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::sign(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return tlx::sign(Xx);
        }
    };

    struct Inverse : KernelBase {
        [[nodiscard]]
        avx2::vec8f operator()(const avx2::vec8f& Xx) const noexcept {
            return avx2::inverse(Xx);
        }
        [[nodiscard]]
        float operator()(const float Xx) const noexcept {
            return 1.f / Xx;
        }
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_HPP