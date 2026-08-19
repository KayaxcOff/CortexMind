//
// Created by muham on 6.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/scalar.hpp"
#include <CortexMind/framework/Engine/Kernels/scalar.hpp>
#include <CortexMind/framework/Engine/Operations/kernel_ops.hpp>
#include <CortexMind/framework/Tools/cast.hpp>
#include <CortexMind/framework/Tools/scratch.hpp>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

namespace {
    template<tlx::extend<ops::KernelBase> OpType>
    void kernel_t(const TensorView& Xx, const float value, TensorView& Xz) {
        switch (Xx.dtype()) {
            case DType::Float32: {
                const auto xx = reinterpret_cast<const float*>(Xx.data());
                const auto xz = reinterpret_cast<float*>(Xz.data());

                kernels::Scalar<OpType>(xx, value, xz, Xx.size());
                break;
            }
            case DType::BFloat16: {
                Scratch x1(Xx.size());
                Scratch x2(Xx.size());

                convert(x1.data(), reinterpret_cast<const tlx::bfloat16 *>(Xx.data()), Xx.size());

                kernels::Scalar<OpType>(x1.data(), value, x2.data(), Xx.size());

                convert(reinterpret_cast<tlx::bfloat16 *>(Xz.data()), x2.data(), Xx.size());
                break;
            }
            case DType::Float16: {
                Scratch x1(Xx.size());
                Scratch x2(Xx.size());

                convert(x1.data(), reinterpret_cast<const tlx::half *>(Xx.data()), Xx.size());

                kernels::Scalar<OpType>(x1.data(), value, x2.data(), Xx.size());

                convert(reinterpret_cast<tlx::half *>(Xz.data()), x2.data(), Xx.size());
                break;
            }
            default:
                WLog(LogLevel::ERROR) << "ScalarOp: unsupported dtype " << as_string(Xx.dtype());
                std::abort();
        }
    }

    template<tlx::extend<ops::KernelBase> OpType>
    void kernel_inplace(TensorView& Xx, const float value) {
        switch (Xx.dtype()) {
            case DType::Float32: {
                const auto xx = reinterpret_cast<float*>(Xx.data());

                kernels::Scalar<OpType>(xx, value,Xx.size());
                break;
            }
            case DType::BFloat16: {
                Scratch x1(Xx.size());

                convert(x1.data(), reinterpret_cast<const tlx::bfloat16 *>(Xx.data()), Xx.size());

                kernels::Scalar<OpType>(x1.data(), value, Xx.size());

                convert(reinterpret_cast<tlx::bfloat16 *>(Xx.data()), x1.data(), Xx.size());
                break;
            }
            case DType::Float16: {
                Scratch x1(Xx.size());

                convert(x1.data(), reinterpret_cast<const tlx::half *>(Xx.data()), Xx.size());

                kernels::Scalar<OpType>(x1.data(), value, Xx.size());

                convert(reinterpret_cast<tlx::half *>(Xx.data()), x1.data(), Xx.size());
                break;
            }
            default:
                WLog(LogLevel::ERROR) << "ScalarOp: unsupported dtype " << as_string(Xx.dtype());
                std::abort();
        }
    }
} //unnamed namespace

void ScalarOp::add(const TensorView &Xx, const float value, TensorView &Xz) {
    kernel_t<ops::Add>(Xx, value, Xz);
}

void ScalarOp::sub(const TensorView &Xx, const float value, TensorView &Xz) {
    kernel_t<ops::Subtract>(Xx, value, Xz);
}

void ScalarOp::mul(const TensorView &Xx, const float value, TensorView &Xz) {
    kernel_t<ops::Multiply>(Xx, value, Xz);
}

void ScalarOp::div(const TensorView &Xx, const float value, TensorView &Xz) {
    kernel_t<ops::Divide>(Xx, value, Xz);
}

void ScalarOp::add(TensorView &Xx, const float value) {
    kernel_inplace<ops::Add>(Xx, value);
}

void ScalarOp::sub(TensorView &Xx, const float value) {
    kernel_inplace<ops::Subtract>(Xx, value);
}

void ScalarOp::mul(TensorView &Xx, const float value) {
    kernel_inplace<ops::Multiply>(Xx, value);
}

void ScalarOp::div(TensorView &Xx, const float value) {
    kernel_inplace<ops::Divide>(Xx, value);
}