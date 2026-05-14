//
// Created by muham on 11.05.2026.
//

#include "CortexMind/framework/Dispatch/scalar_operations.hpp"
#include <CortexMind/framework/Engine/AVX2/scalar.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/scalar.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::disp;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

Scalar::Scalar(const DeviceType _d_type) : d_type(_d_type) {}

Scalar::~Scalar() = default;

void Scalar::SetDevice(const DeviceType _d_type) {
    CXM_WARN(this->d_type == _d_type, "You already using " + as_string(this->d_type) + " device");
    this->d_type = _d_type;
}

void Scalar::add(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");

    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::add(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::add(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::add(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Scalar::sub(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");

    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::sub(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::sub(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::sub(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Scalar::mul(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");

    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::mul(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::mul(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::mul(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Scalar::div(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xz == nullptr, "Storages/Storage are/is null");
    CXM_ASSERT(value == 0, "Scalar value must be non-zero for division");
    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::div(Xx->data(), value, Xz->data(), N);
        } else {
            cuda::ScalarKernel::div(Xx->data(), value, Xz->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::div(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Scalar::add(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(Xx == nullptr, "Storage is null");

    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::add(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::add(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::add(Xx->data(), value, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Scalar::sub(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(Xx == nullptr, "Storage is null");

    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::sub(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::sub(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::sub(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Scalar::mul(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(Xx == nullptr, "Storage is null");

    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::mul(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::mul(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::mul(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Scalar::div(TensorStorage *Xx, const f32 value, const size_t N) {
    CXM_ASSERT(Xx == nullptr, "Storage is null");
    CXM_ASSERT(value == 0, "Scalar value cannot be zero for division");
    CXM_ASSERT(N == 0, "Number of tensors cannot be zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            avx2::ScalarOp::div(Xx->data(), value, N);
        } else {
            cuda::ScalarKernel::div(Xx->data(), value, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::ScalarOp::div(Xx->data(), value, Xz->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}