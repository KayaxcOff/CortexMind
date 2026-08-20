//
// Created by muham on 7.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/wise.hpp"
#include <CortexMind/framework/Engine/Kernels/unary.hpp>
#include <CortexMind/framework/Engine/Kernels/scalar.hpp>
#include <CortexMind/framework/Engine/Operations/kernel_ops.hpp>
#include <CortexMind/framework/Tools/cast.hpp>
#include <CortexMind/framework/Tools/scratch.hpp>
#include <algorithm>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

namespace {
    template<tlx::extend<ops::KernelBase> OpType>
    void kernel_t(const TensorView& Xx, TensorView& Xz) {
        switch (Xx.dtype()) {
            case DType::Float32: {
                const auto xx = reinterpret_cast<const float*>(Xx.data());
                const auto xz = reinterpret_cast<float*>(Xz.data());

                kernels::Unary<OpType>(xx, xz, Xx.size());
                break;
            }
            case DType::BFloat16: {
                Scratch x1(Xx.size());
                Scratch x2(Xx.size());

                convert(x1.data(), reinterpret_cast<const tlx::bfloat16 *>(Xx.data()), Xx.size());

                kernels::Unary<OpType>(x1.data(),x2.data(), Xx.size());

                convert(reinterpret_cast<tlx::bfloat16 *>(Xz.data()), x2.data(), Xx.size());
                break;
            }
            case DType::Float16: {
                Scratch x1(Xx.size());
                Scratch x2(Xx.size());

                convert(x1.data(), reinterpret_cast<const tlx::half *>(Xx.data()), Xx.size());

                kernels::Unary<OpType>(x1.data(), x2.data(), Xx.size());

                convert(reinterpret_cast<tlx::half *>(Xz.data()), x2.data(), Xx.size());
                break;
            }
            default:
                WLog(LogLevel::ERROR) << "ScalarOp: unsupported dtype " << as_string(Xx.dtype());
                std::abort();
        }
    }
} //unnamed namespace

void wise::square(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Square>(Xx, Xz);
}

void wise::pow(const TensorView &Xx, const float value, TensorView &Xz) {
    switch (Xx.dtype()) {
        case DType::Float32: {
            const auto xx = reinterpret_cast<const float*>(Xx.data());
            const auto xz = reinterpret_cast<float*>(Xz.data());

            kernels::Scalar<ops::Power>(xx, value, xz, Xx.size());
        }
        case DType::BFloat16: {
            Scratch x1(Xx.size());
            Scratch x2(Xx.size());

            convert(x1.data(), reinterpret_cast<const tlx::bfloat16 *>(Xx.data()), Xx.size());

            kernels::Scalar<ops::Power>(x1.data(), value, x2.data(), Xx.size());

            convert(reinterpret_cast<tlx::bfloat16 *>(Xz.data()), x2.data(), Xx.size());
        }
        case DType::Float16: {
            Scratch x1(Xx.size());
            Scratch x2(Xx.size());

            convert(x1.data(), reinterpret_cast<const tlx::half *>(Xx.data()), Xx.size());

            kernels::Scalar<ops::Power>(x1.data(), value, x2.data(), Xx.size());

            convert(reinterpret_cast<tlx::half *>(Xz.data()), x2.data(), Xx.size());
        }
        default:
            std::abort();
    }
}

void wise::pow(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {

}

void wise::sqrt(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Sqrt>(Xx, Xz);
}

void wise::rsqrt(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::RSqrt>(Xx, Xz);
}

void wise::log(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Log>(Xx, Xz);
}

void wise::exp(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Exp>(Xx, Xz);
}

void wise::erf(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Erf>(Xx, Xz);
}

void wise::sin(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Sin>(Xx, Xz);
}

void wise::cos(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Cos>(Xx, Xz);
}

void wise::abs(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Abs>(Xx, Xz);
}

void wise::neg(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Neg>(Xx, Xz);
}

void wise::rcp(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Rcp>(Xx, Xz);
}

void wise::inverse(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Inverse>(Xx, Xz);
}

void wise::sign(const TensorView &Xx, TensorView &Xz) {
    kernel_t<ops::Sign>(Xx, Xz);
}

void wise::lerp(const float *Xx, const float value1, const float value2, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto v1 = set1(value1);
    const auto v2 = set1(value2);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::lerp(loadu(Xx + i), v1, v2));
    }
    for (; i < N; ++i) {
        Xz[i] = tlx::lerp(Xx[i], value1, value2);
    }
}

void wise::clamp(const float *Xx, const float min, const float max, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    const auto mn = set1(min);
    const auto mx = set1(max);
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::clamp(loadu(Xx + i), mn, mx));
    }
    for (; i < N; ++i) {
        Xz[i] = std::clamp(Xx[i], min, max);
    }
}

void wise::gather(const float *Xx, const std::int32_t *Xy, float* Xz, const std::size_t N) {
    std::size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        const auto indices = loadu(Xy + i);

        storeu(Xz + i, avx2::gather(Xx, indices));
    }

    for (; i < N; ++i) {
        Xz[i] = Xx[Xy[i]];
    }
}

void wise::gather(const std::int32_t *Xx, const std::int32_t *Xy, std::int32_t* Xz, const std::size_t N) {
    std::size_t i = 0;

    for (; i + 8 <= N; i += 8) {
        const auto indices = loadu(Xy + i);

        storeu(Xz + i, avx2::gather(Xx, indices));
    }

    for (; i < N; ++i) {
        Xz[i] = Xx[Xy[i]];
    }
}