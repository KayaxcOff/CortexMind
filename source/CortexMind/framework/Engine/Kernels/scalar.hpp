//
// Created by muham on 19.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_SCALAR_HPP

#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/Operations/base.hpp>
#include <tlx/concepts.hpp>

namespace cortex::_fw::kernels {
    template<tlx::extend<ops::KernelBase> OpType>
    void Scalar(const float* __restrict Xx, float value, float* __restrict Xz, const std::size_t N) {
        OpType op;

        std::size_t i = 0;
        avx2::vec8f r1 = avx2::set1(value);
        for (; i + 8 <= N; i += 8) {
            avx2::vec8f r0 = avx2::loadu(Xx + i);
            avx2::vec8f output = op(r0, r1);
            avx2::storeu(Xz + i, output);
        }
        for (; i < N; i += 8) {
            Xz[i] = op(Xx[i], value);
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    void Scalar(float* Xx, float value, const std::size_t N) {
        OpType op;

        std::size_t i = 0;
        avx2::vec8f r1 = avx2::set1(value);
        for (; i + 8 <= N; i += 8) {
            avx2::vec8f r0 = avx2::loadu(Xx + i);
            avx2::vec8f output = op(r0, r1);
            avx2::storeu(Xx + i, output);
        }
        for (; i < N; i += 8) {
            Xx[i] = op(Xx[i], value);
        }
    }
} //namespace cortex::_fw::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_KERNELS_SCALAR_HPP