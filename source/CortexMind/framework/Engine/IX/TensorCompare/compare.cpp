//
// Created by muham on 7.06.2026.
//

#include "CortexMind/framework/Engine/IX/TensorCompare/compare.hpp"
#include <CortexMind/framework/Engine/AVX2/compare.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/compare.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void TensorCompare::gt(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz) noexcept {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One Storage's device is " + as_string(Xx->device()) + " other Storage's device is " + as_string(Xy->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::compare::gt(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        } else {
            cuda::CompareTo::greater(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::compare::gt(Xx->data(), Xy->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorCompare::lt(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz) noexcept {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One Storage's device is " + as_string(Xx->device()) + " other Storage's device is " + as_string(Xy->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::compare::lt(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        } else {
            cuda::CompareTo::less(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::compare::lt(Xx->data(), Xy->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorCompare::eq(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz) noexcept {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One Storage's device is " + as_string(Xx->device()) + " other Storage's device is " + as_string(Xy->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::compare::eq(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        } else {
            cuda::CompareTo::equal(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::compare::gt(Xx->data(), Xy->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorCompare::ge(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz) noexcept {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One Storage's device is " + as_string(Xx->device()) + " other Storage's device is " + as_string(Xy->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::compare::ge(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        } else {
            cuda::CompareTo::greater_eq(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::compare::ge(Xx->data(), Xy->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorCompare::le(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz) noexcept {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One Storage's device is " + as_string(Xx->device()) + " other Storage's device is " + as_string(Xy->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::compare::le(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        } else {
            cuda::CompareTo::less_eq(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::compare::le(Xx->data(), Xy->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorCompare::neq(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz) noexcept {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One Storage's device is " + as_string(Xx->device()) + " other Storage's device is " + as_string(Xy->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::compare::neq(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        } else {
            //cuda::CompareTo::equal(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::compare::neq(Xx->data(), Xy->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}