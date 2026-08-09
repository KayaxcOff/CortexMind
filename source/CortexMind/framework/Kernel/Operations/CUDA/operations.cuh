//
// Created by muham on 9.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_KERNEL_OPERATIONS_CUDA_OPERATIONS_CUH
#define CORTEXMIND_FRAMEWORK_KERNEL_OPERATIONS_CUDA_OPERATIONS_CUH

#include <CortexMind/framework/Kernel/Operations/CUDA/base.cuh>

namespace cortex::_fw::ops {
    template<tlx::arithmetic_like T>
    struct addition : kernel_base<T> {
        addition() : kernel_base<T>("addition") {}

        [[nodiscard]]
        T operator()(const T Xx, const T Xy) const noexcept {
            return Xx + Xy;
        }
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_KERNEL_OPERATIONS_CUDA_OPERATIONS_CUH