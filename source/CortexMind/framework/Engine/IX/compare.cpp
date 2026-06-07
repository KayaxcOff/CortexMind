//
// Created by muham on 21.05.2026.
//

#include "CortexMind/framework/Engine/IX/compare.hpp"
#include <CortexMind/framework/Engine/AVX2/reduce.hpp>
#include <CortexMind/framework/Engine/AVX2/wise.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/compare.cuh>
    #include <CortexMind/framework/Memory/transform.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <vector>

using namespace cortex::_fw::ix;
using namespace cortex::_fw;

void CompareTo::greater(const TensorStorage *Xx, const TensorStorage* Xy, TensorStorage* Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is null");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One device is " + as_string(Xx->device()) + " and other one is " + as_string(Xy->device()));

    auto dev = sys::DeviceType::kHOST;
    if (Xx->device() == Xy->device()) [[likely]] {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == sys::DeviceType::kHOST) {
            avx2::(Xx->data(), Xy->data(), Xz->data(), N);
        } else {
            cuda::CompareTo::greater(Xx->data(), Xy->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::greater(Xx->data(), Xy->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

void CompareTo::less(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is null");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One device is " + as_string(Xx->device()) + " and other one is " + as_string(Xy->device()));

    auto dev = sys::DeviceType::kHOST;
    if (Xx->device() == Xy->device()) [[likely]] {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == sys::DeviceType::kHOST) {
            avx2::wise::less(Xx->data(), Xy->data(), Xz->data(), N);
        } else {
            cuda::CompareTo::less(Xx->data(), Xy->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::less(Xx->data(), Xy->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

void CompareTo::greater_eq(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is null");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One device is " + as_string(Xx->device()) + " and other one is " + as_string(Xy->device()));

    auto dev = sys::DeviceType::kHOST;
    if (Xx->device() == Xy->device()) [[likely]] {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == sys::DeviceType::kHOST) {
            avx2::wise::greater_eq(Xx->data(), Xy->data(), Xz->data(), N);
        } else {
            cuda::CompareTo::greater_eq(Xx->data(), Xy->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::greater_eq(Xx->data(), Xy->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

void CompareTo::less_eq(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is null");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One device is " + as_string(Xx->device()) + " and other one is " + as_string(Xy->device()));

    auto dev = sys::DeviceType::kHOST;
    if (Xx->device() == Xy->device()) [[likely]] {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == sys::DeviceType::kHOST) {
            avx2::wise::less_eq(Xx->data(), Xy->data(), Xz->data(), N);
        } else {
            cuda::CompareTo::less_eq(Xx->data(), Xy->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::less_eq(Xx->data(), Xy->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

bool CompareTo::equal(const TensorStorage *Xx, const TensorStorage *Xy, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is null");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "One device is " + as_string(Xx->device()) + " and other one is " + as_string(Xy->device()));

    auto dev = sys::DeviceType::kHOST;
    if (Xx->device() == Xy->device()) [[likely]] {
        dev = Xx->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == sys::DeviceType::kCUDA) {
            std::vector<f32> hx(N), hy(N);
            sys::transform::download(hy.data(), Xy->data(), N);
            sys::transform::download(hx.data(), Xx->data(), N);
            return avx2::reduce::equal(hx.data(), hy.data(), N);
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE
    return avx2::reduce::equal(Xx->data(), Xy->data(), N);
}
