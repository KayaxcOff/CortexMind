//
// Created by muham on 12.05.2026.
//

#include "CortexMind/framework/Dispatch/elem_wise_operations.hpp"
#include <CortexMind/framework/Engine/AVX2/wise.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/element_wise.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::disp;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

ElementWise::ElementWise(const DeviceType _d_type) : d_type(_d_type) {}

ElementWise::~ElementWise() = default;

void ElementWise::SetDevice(const DeviceType _d_type) {
    this->d_type = _d_type;
}

void ElementWise::pow(const TensorStorage *Xx, const f32 exp, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number of tensor element must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::wise::pow(Xx->data(), exp, Xz->data(), N);
        } else {
            cuda::ElementWise::pow(Xx->data(), exp, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::pow(Xx->data(), exp, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::sqrt(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number of tensor element must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::wise::sqrt(Xx->data(),Xz->data(), N);
        } else {
            cuda::ElementWise::sqrt(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::sqrt(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::exp(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number of tensor element must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::wise::exp(Xx->data(),Xz->data(), N);
        } else {
            cuda::ElementWise::exp(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::exp(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::log(const TensorStorage *Xx, TensorStorage *Xz, size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number of tensor element must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::wise::log(Xx->data(),Xz->data(), N);
        } else {
            cuda::ElementWise::log(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::log(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void ElementWise::abs(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number of tensor element must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::wise::abs(Xx->data(),Xz->data(), N);
        } else {
            cuda::ElementWise::abs(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::wise::abs(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}