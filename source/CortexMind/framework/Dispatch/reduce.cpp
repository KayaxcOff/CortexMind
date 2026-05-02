//
// Created by muham on 1.05.2026.
//

#include "CortexMind/framework/Dispatch/reduce.hpp"
#include <CortexMind/core/Engine/AVX2/reduce.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/reduce.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/runtime/macros.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw::txl;
using namespace cortex::_fw;

ReduceManager::ReduceManager() : d_type(deviceType::host), max_dim(CXM_THRESHOLD) {}

ReduceManager::~ReduceManager() = default;

void ReduceManager::SetDevice(const deviceType _d_type) {
    this->d_type = _d_type;
}

f32 ReduceManager::sum(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::sum(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.sum(Xx->data(), Xx->size());
        #else
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return 0.0f;
}

f32 ReduceManager::mean(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::mean(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.mean(Xx->data(), Xx->size());
        #else
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return 0.0f;
}

f32 ReduceManager::var(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::var(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.var(Xx->data(), Xx->size());
        #else
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return 0.0f;
}

f32 ReduceManager::stdv(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::std(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.std(Xx->data(), Xx->size());
        #else
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return 0.0f;
}

f32 ReduceManager::min(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::min(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.min(Xx->data(), Xx->size());
        #else //#if CXM_IS_CUDA_AVAILABLE
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
    return 0.0f;
}

f32 ReduceManager::max(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::max(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.max(Xx->data(), Xx->size());
        #else //#if CXM_IS_CUDA_AVAILABLE
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE #else
    }
    return 0.0f;
}

f32 ReduceManager::norm1(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::norm1(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.norm1(Xx->data(), Xx->size());
        #else //#if CXM_IS_CUDA_AVAILABLE
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE #else
    }
    return 0.0f;
}

f32 ReduceManager::norm2(const TensorStorage *Xx) const {
    if (this->d_type == deviceType::host) {
        return avx2::reduce::norm2(Xx->data(), Xx->size());
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.norm2(Xx->data(), Xx->size());
        #else //#if CXM_IS_CUDA_AVAILABLE
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE #else
    }
    return 0.0f;
}

f32 ReduceManager::dot(const TensorStorage *Xx, const TensorStorage *Xy) const {
    size_t min_size = Xx->size() < Xy->size() ? Xx->size() : Xy->size();

    if (this->d_type == deviceType::host) {
        return avx2::reduce::dot(Xx->data(), Xy->data(), min_size);
    } else if (this->d_type == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp op;
            return op.dot(Xx->data(), Xy->data(), min_size);
        #else //#if CXM_IS_CUDA_AVAILABLE
            return 0.0f;
        #endif //#if CXM_IS_CUDA_AVAILABLE #else
    }
    return 0.0f;
}