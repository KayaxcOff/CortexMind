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

TensorScalarExecutor::TensorScalarExecutor() : d_type(deviceType::host), max_dim(CXM_MAX_ITEMS) {}

TensorScalarExecutor::~TensorScalarExecutor() = default;

void TensorScalarExecutor::SetDevice(const deviceType _d_type) {
    this->d_type = _d_type;
}

void TensorScalarExecutor::addition(const TensorStorage *Xx, const f32 value, TensorStorage *Xz) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::add(Xx->data(), value, Xz->data(), Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::add(Xx->data(), value, Xz->data(), Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::add(Xx->data(), value, Xz->data(), Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalarExecutor::subtraction(const TensorStorage *Xx, const f32 value, TensorStorage *Xz) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::sub(Xx->data(), value, Xz->data(), Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::sub(Xx->data(), value, Xz->data(), Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::sub(Xx->data(), value, Xz->data(), Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalarExecutor::multiply(const TensorStorage *Xx, const f32 value, TensorStorage *Xz) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::mul(Xx->data(), value, Xz->data(), Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::mul(Xx->data(), value, Xz->data(), Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::mul(Xx->data(), value, Xz->data(), Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalarExecutor::division(const TensorStorage *Xx, const f32 value, TensorStorage *Xz) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::div(Xx->data(), value, Xz->data(), Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::div(Xx->data(), value, Xz->data(), Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::div(Xx->data(), value, Xz->data(), Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalarExecutor::addition(TensorStorage *Xx, const f32 value) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::add(Xx->data(), value, Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::add(Xx->data(), value, Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::add(Xx->data(), value, Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalarExecutor::subtraction(TensorStorage *Xx, const f32 value) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::sub(Xx->data(), value, Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::sub(Xx->data(), value, Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::sub(Xx->data(), value, Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalarExecutor::multiply(TensorStorage *Xx, const f32 value) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::mul(Xx->data(), value, Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::mul(Xx->data(), value, Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::mul(Xx->data(), value, Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

void TensorScalarExecutor::division(TensorStorage *Xx, const f32 value) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > this->max_dim) {
            avx2::ScalarOp::div(Xx->data(), value, Xx->size());
        } else if (Xx->size() <= this->max_dim) {
            stl::Scalar::div(Xx->data(), value, Xx->size());
        }
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ScalarKernel::div(Xx->data(), value, Xx->size());
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}
