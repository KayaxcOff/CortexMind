//
// Created by muham on 16.05.2026.
//

#include "CortexMind/framework/Engine/IX/element_wise.hpp"
#include <CortexMind/framework/Engine/AVX2/wise.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/element_wise.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

ElementWise::ElementWise() = default;

ElementWise::~ElementWise() = default;

void ElementWise::pow(const TensorStorage *Xx, const f32 exp, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::pow(Xx->data(), exp, Xz->data(), N);
        } else {
            cuda::ElementWise::pow(Xx->data(), exp, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::pow(Xx->data(), exp, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::sqrt(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::sqrt(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::sqrt(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sqrt(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::log(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() != Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::log(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::log(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::log(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::exp(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() != Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::exp(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::exp(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::exp(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::abs(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() != Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::abs(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::abs(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::abs(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::sin(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() != Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::sin(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::sin(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sin(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::cos(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::cos(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::cos(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::cos(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::rsqrt(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::rsqrt(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::rsqrt(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::rsqrt(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::sign(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::sign(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::sign(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sign(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::neg(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::wise::neg(Xx->data(), Xz->data(), N);
        } else {
            cuda::ElementWise::neg(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::neg(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}
