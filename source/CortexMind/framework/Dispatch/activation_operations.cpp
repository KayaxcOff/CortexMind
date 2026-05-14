//
// Created by muham on 14.05.2026.
//

#include "CortexMind/framework/Dispatch/activation_operations.hpp"
#include <CortexMind/framework/Engine/AVX2/activation.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/activation.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::disp;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

Activation::Activation(const DeviceType _d_type) : d_type(_d_type) {}

Activation::~Activation() = default;

void Activation::SetDevice(const DeviceType _d_type) {
    this->d_type = _d_type;
}

void Activation::relu(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::relu(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::relu(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::relu(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::leaky_relu(const TensorStorage *Xx, TensorStorage *Xz, const size_t N, const f32 alpha) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::leaky_relu(Xx->data(), Xz->data(), N, alpha);
        } else {
            cuda::Activation::leaky_relu(Xx->data(), Xz->data(), N, alpha);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::leaky_relu(Xx->data(), Xz->data(), N, alpha);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::sigmoid(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::sigmoid(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::sigmoid(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::sigmoid(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::sigmoid_fast(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::sigmoid_fast(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::sigmoid_fast(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::sigmoid(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::tanh(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::tanh(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::tanh(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::tanh(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::gelu(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::gelu(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::gelu(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::gelu(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::gelu_exact(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::gelu_exact(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::gelu_exact(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::gelu_exact(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::silu(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::silu(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::silu(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::silu(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::silu_fast(const TensorStorage *Xx, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::silu_fast(Xx->data(), Xz->data(), N);
        } else {
            cuda::Activation::silu_fast(Xx->data(), Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::silu_fast(Xx->data(), Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::swish(const TensorStorage *Xx, TensorStorage *Xz, const size_t N, const f32 beta) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::swish(Xx->data(), Xz->data(), N, beta);
        } else {
            cuda::Activation::swish(Xx->data(), Xz->data(), N, beta);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::swish(Xx->data(), Xz->data(), N, beta);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Activation::swish_fast(const TensorStorage *Xx, TensorStorage *Xz, const size_t N, const f32 beta) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(N == 0, "Number element of tensor must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::Activation::swish_fast(Xx->data(), Xz->data(), N, beta);
        } else {
            cuda::Activation::swish_fast(Xx->data(), Xz->data(), N, beta);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::Activation::swish_fast(Xx->data(), Xz->data(), N, beta);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}
