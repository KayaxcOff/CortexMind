//
// Created by muham on 20.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_UNARY_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_UNARY_HPP

#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/Operations/base.hpp>
#include <tlx/concepts.hpp>

namespace cortex::_fw::kernels {
    template<tlx::extend<ops::KernelBase> OpType>
    void Unary(const float* __restrict Xx, float* __restrict Xz, const std::size_t N) {
        OpType op;

        std::size_t i = 0;
        for (; i + 8 < N; i += 8) {
            avx2::vec8f r0 = avx2::loadu(Xx + i);
            avx2::storeu(Xz + i, op(r0));
        }
        for (; i < N; ++i) {
            Xz[i] = op(Xz[i]);
        }
    }
} //namespace cortex::_fw::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_UNARY_HPP