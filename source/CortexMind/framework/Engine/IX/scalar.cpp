//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Engine/IX/scalar.hpp"
#include <CortexMind/framework/Engine/AVX2/scalar.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/scalar.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

ScalarOp::ScalarOp() = default;

ScalarOp::~ScalarOp() = default;

void ScalarOp::add(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xz->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::add(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::add(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::add(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ScalarOp::sub(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xz->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::sub(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::sub(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::sub(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ScalarOp::mul(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xz->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::mul(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::mul(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::mul(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ScalarOp::div(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");
    CXM_ASSERT(value == 0, "Scalar value can't be zero for division");

    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xz->device()) {
        dev = Xz->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::div(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::div(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::div(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ScalarOp::add(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    const auto dev = Xx->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::add(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::add(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::add(Xx->data(), value, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ScalarOp::sub(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    const auto dev = Xx->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::sub(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::sub(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::sub(Xx->data(), value, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ScalarOp::mul(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    const auto dev = Xx->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::mul(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::mul(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::mul(Xx->data(), value, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ScalarOp::div(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");
    CXM_ASSERT(value == 0, "Scalar value can't be zero for division");

    const auto dev = Xx->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::ScalarOp::div(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::div(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::div(Xx->data(), value, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}