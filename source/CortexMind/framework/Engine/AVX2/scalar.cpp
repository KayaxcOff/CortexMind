//
// Created by muham on 6.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/scalar.hpp"
#include <CortexMind/framework/Engine/Kernels/scalar.hpp>
#include <CortexMind/framework/Engine/Operations/kernel_ops.hpp>

using namespace cortex::_fw::avx2;

void ScalarOp::add(const TensorView &Xx, const float value, TensorView &Xz) {
    const auto xf = reinterpret_cast<const float*>(Xx.data());
    const auto zf = reinterpret_cast<float*>(Xz.data());

    kernels::Scalar<ops::Add>(xf, value, zf, Xx.size());
}

void ScalarOp::sub(const TensorView &Xx, const float value, TensorView &Xz) {
    const auto xf = reinterpret_cast<const float*>(Xx.data());
    const auto zf = reinterpret_cast<float*>(Xz.data());

    kernels::Scalar<ops::Subtract>(xf, value, zf, Xx.size());
}

void ScalarOp::mul(const TensorView &Xx, const float value, TensorView &Xz) {
    const auto xf = reinterpret_cast<const float*>(Xx.data());
    const auto zf = reinterpret_cast<float*>(Xz.data());

    kernels::Scalar<ops::Multiply>(xf, value, zf, Xx.size());
}

void ScalarOp::div(const TensorView &Xx, const float value, TensorView &Xz) {
    const auto xf = reinterpret_cast<const float*>(Xx.data());
    const auto zf = reinterpret_cast<float*>(Xz.data());

    kernels::Scalar<ops::Divide>(xf, value, zf, Xx.size());
}

void ScalarOp::add(TensorView &Xx, const float value) {
    const auto xf = reinterpret_cast<float*>(Xx.data());

    kernels::Scalar<ops::Add>(xf, value,Xx.size());
}

void ScalarOp::sub(TensorView &Xx, const float value) {
    const auto xf = reinterpret_cast<float*>(Xx.data());

    kernels::Scalar<ops::Subtract>(xf, value,Xx.size());
}

void ScalarOp::mul(TensorView &Xx, const float value) {
    const auto xf = reinterpret_cast<float*>(Xx.data());

    kernels::Scalar<ops::Multiply>(xf, value,Xx.size());
}

void ScalarOp::div(TensorView &Xx, const float value) {
    const auto xf = reinterpret_cast<float*>(Xx.data());

    kernels::Scalar<ops::Divide>(xf, value,Xx.size());
}