//
// Created by muham on 1.05.2026.
//

#include "CortexMind/framework/Dispatch/activation.hpp"
#include <CortexMind/core/Engine/AVX2/activation.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/activation.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/core/Engine/STD/activation.hpp>
#include <CortexMind/runtime/macros.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw::txl;
using namespace cortex::_fw;

ActivationManager::ActivationManager() : d_type(deviceType::host), max_dim(CXM_MAX_ITEMS) {}

ActivationManager::~ActivationManager() = default;

void ActivationManager::SetDevice(const deviceType _d_type) {
    this->d_type = _d_type;
}

void ActivationManager::ReLU(const TensorStorage *Xx, TensorStorage *Xz) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::Activation::relu(Xx->data(), Xz->data(), Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::ActivationOp::relu(Xx->data(), Xz->data(), Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Activation::relu(Xx->data(), Xz->data(), Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void ActivationManager::LeakyReLU(const TensorStorage *Xx, const f32 alpha, TensorStorage *Xz) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::Activation::leaky_relu(Xx->data(), Xz->data(), Xx->size(), alpha);
        } else if (Xx->size() <= this->max_dim) {
            stl::ActivationOp::leaky_relu(Xx->data(), alpha, Xz->data(), Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Activation::leaky_relu(Xx->data(), Xz->data(), Xx->size(), alpha);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void ActivationManager::Sigmoid(const TensorStorage *Xx, TensorStorage *Xz) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::Activation::sigmoid(Xx->data(), Xz->data(), Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::ActivationOp::sigmoid(Xx->data(), Xz->data(), Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::Activation::sigmoid(Xx->data(), Xz->data(), Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}
