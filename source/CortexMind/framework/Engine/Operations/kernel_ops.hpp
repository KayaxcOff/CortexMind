//
// Created by muham on 19.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_HPP

#include <CortexMind/framework/Engine/Operations/base.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>

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
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_ENGINE_OPERATIONS_KERNEL_OPS_HPP