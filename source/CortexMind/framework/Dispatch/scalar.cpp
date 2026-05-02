//
// Created by muham on 29.04.2026.
//

#include "CortexMind/framework/Dispatch/scalar.hpp"
#include <CortexMind/core/Engine/AVX2/scalar.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/scalar.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/core/Engine/STD/scalar.hpp>
#include <CortexMind/runtime/macros.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw::txl;
using namespace cortex::_fw;

TensorScalar::TensorScalar(const deviceType _d_type) : d_type(_d_type), max_dim(CXM_THRESHOLD) {}

TensorScalar::~TensorScalar() = default;

void TensorScalar::add(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::add(Xx->data(), value, Xz->data(), N);
        } else {
            stl::Scalar::add(Xx->data(), value, Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::add(Xx->data(), value, Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalar::sub(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::sub(Xx->data(), value, Xz->data(), N);
        } else {
            stl::Scalar::sub(Xx->data(), value, Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::sub(Xx->data(), value, Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalar::mul(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::mul(Xx->data(), value, Xz->data(), N);
        } else {
            stl::Scalar::mul(Xx->data(), value, Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::mul(Xx->data(), value, Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalar::div(const TensorStorage *Xx, const f32 value, TensorStorage *Xz, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::div(Xx->data(), value, Xz->data(), N);
        } else {
            stl::Scalar::div(Xx->data(), value, Xz->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::div(Xx->data(), value, Xz->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalar::add(TensorStorage *Xx, const f32 value, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::add(Xx->data(), value, N);
        } else {
            stl::Scalar::add(Xx->data(), value, N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::add(Xx->data(), value, N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalar::sub(TensorStorage *Xx, const f32 value, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::sub(Xx->data(), value, N);
        } else {
            stl::Scalar::sub(Xx->data(), value, N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::sub(Xx->data(), value, N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalar::mul(TensorStorage *Xx, const f32 value, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::mul(Xx->data(), value, N);
        } else {
            stl::Scalar::mul(Xx->data(), value, N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::mul(Xx->data(), value, N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalar::div(TensorStorage *Xx, const f32 value, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            avx2::ScalarOp::div(Xx->data(), value, N);
        } else {
            stl::Scalar::div(Xx->data(), value, N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::div(Xx->data(), value, N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}
