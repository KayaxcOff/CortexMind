//
// Created by muham on 2.05.2026.
//

#include "CortexMind/framework/Dispatch/wise.hpp"
#include <CortexMind/core/Engine/AVX2/wise.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/elem_wise.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/core/Engine/STD/element_wise.hpp>
#include <CortexMind/runtime/macros.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw::txl;
using namespace cortex::_fw;

Wise::Wise(const deviceType _d_type) : d_type(_d_type), max_dim(CXM_THRESHOLD) {}

Wise::~Wise() = default;

void Wise::pow(const TensorStorage *Xx, const f32 exp, TensorStorage *Xz, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::wise::pow(Xx->data(), exp, Xz->data(), N);
        } else {
            stl::Element::pow(Xx->data(), exp, Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ElementWise::pow(Xx->data(), exp, Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void Wise::sqrt(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::wise::square(Xx->data(), Xz->data(), N);
        } else {
            stl::Element::sqrt(Xx->data(), Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ElementWise::sqrt(Xx->data(), Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void Wise::log(const TensorStorage *Xx, TensorStorage *Xz, size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::wise::log(Xx->data(), Xz->data(), N);
        } else {
            stl::Element::log(Xx->data(), Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ElementWise::log(Xx->data(), Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void Wise::exp(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::wise::exp(Xx->data(), Xz->data(), N);
        } else {
            stl::Element::exp(Xx->data(), Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ElementWise::exp(Xx->data(), Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}
