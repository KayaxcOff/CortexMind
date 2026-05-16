//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Engine/IX/reduce.hpp"
#include <CortexMind/framework/Engine/AVX2/reduce.hpp>
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <limits>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

reduce::reduce() = default;

reduce::~reduce() = default;

#if CXM_IS_CUDA_AVAILABLE
    static cuda::ReduceOp op;
#endif //#if CXM_IS_CUDA_AVAILABLE

f32 reduce::sum(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::sum(x->data(), N);
        } else {
            output = op.sum(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::sum(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::mean(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::mean(x->data(), N);
        } else {
            output = op.mean(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::mean(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::var(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::var(x->data(), N);
        } else {
            output = op.var(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::var(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::stdv(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::std(x->data(), N);
        } else {
            output = op.stdv(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::std(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::min(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::min(x->data(), N);
        } else {
            output = op.min(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::min(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::max(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::max(x->data(), N);
        } else {
            output = op.max(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::max(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::norm1(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::norm1(x->data(), N);
        } else {
            output = op.norm1(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::norm1(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::norm2(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::norm2(x->data(), N);
        } else {
            output = op.norm2(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::norm2(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::dot(const TensorStorage *Xx, const TensorStorage *Xy, const size_t N) {
    CXM_ASSERT(Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(Xy->isValid(), "Input Storage is null");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xy->device()));

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xy->device()) {
        dev = Xy->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::dot(Xx->data(), Xy->data(), N);
        } else {
            output = op.dot(Xx->data(), Xy->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::dot(Xx->data(), Xy->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}