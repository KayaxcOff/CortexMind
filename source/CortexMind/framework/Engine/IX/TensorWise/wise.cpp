//
// Created by muham on 6.06.2026.
//

#include "CortexMind/framework/Engine/IX/TensorWise/wise.hpp"
#include <CortexMind/framework/Engine/AVX2/wise.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/element_wise.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void TensorWise::pow(const TensorStorage *Xx, const f32 exp, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::pow(Xx->data(), exp, Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::pow(Xx->data(), exp, Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::pow(Xx->data(), exp, Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::sqrt(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::sqrt(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::sqrt(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sqrt(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::rsqrt(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::rsqrt(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::rsqrt(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::rsqrt(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::square(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::sqrt(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::sqrt(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sqrt(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::exp(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::exp(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::exp(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::exp(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::exp2(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::exp2(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::exp2(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::exp2(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::exp10(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::exp10(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::exp10(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::exp10(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::log(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::log(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::log(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::log(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::log2(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::log2(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::log2(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::log2(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::log10(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::log10(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::log10(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::log10(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::erf(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::erf(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::erf(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::erf(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::sin(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::sin(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::sin(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sinXx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::cos(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::cos(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::cos(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::cos(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::tan(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::tan(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::tan(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::tan(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::cot(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::cot(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::cot(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::cot(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::abs(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::abs(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::abs(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::abs(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::neg(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::neg(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::neg(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::neg(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::sign(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::sign(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::sign(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sign(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::reciprocal(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            //avx2::wise::inv(Xx->data(), Xz->data(), Xx->size());
        } else {
            //cuda::ElementWise::inv(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        //avx2::wise::sqrt(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorWise::clamp(const TensorStorage *Xx, const f32 min, const f32 max, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::clamp(Xx->data(), min, max, Xz->data(), Xx->size());
        } else {
            cuda::ElementWise::clamp(Xx->data(), min, max, Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::clamp(Xx->data(), min, max, Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}
