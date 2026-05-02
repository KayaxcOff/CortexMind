//
// Created by muham on 1.05.2026.
//

#include "CortexMind/framework/Dispatch/reduce.hpp"
#include <CortexMind/core/Engine/AVX2/reduce.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/reduce.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/core/Engine/STD/reduce.hpp>
#include <CortexMind/runtime/macros.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw::txl;
using namespace cortex::_fw;

ReductionOps::ReductionOps(const deviceType _d_type) : d_type(_d_type), max_dim(CXM_THRESHOLD) {}

ReductionOps::~ReductionOps() = default;

void ReductionOps::SetDevice(const deviceType _d_type) {
    this->d_type = _d_type;
}

f32 ReductionOps::sum(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::sum(Xx->data(), N);
        } else {
            return stl::Reduce::sum(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.sum(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::mean(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::mean(Xx->data(), N);
        } else {
            return stl::Reduce::mean(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.mean(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::var(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::var(Xx->data(), N);
        } else {
            return stl::Reduce::variance(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.var(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::stdv(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::std(Xx->data(), N);
        } else {
            return stl::Reduce::standard_deviation(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.std(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::min(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::min(Xx->data(), N);
        } else {
            return stl::Reduce::min(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.min(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::max(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::max(Xx->data(), N);
        } else {
            return stl::Reduce::max(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.max(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::norm1(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::norm1(Xx->data(), N);
        } else {
            return stl::Reduce::norm1(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.norm1(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::norm2(const TensorStorage *Xx, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::norm2(Xx->data(), N);
        } else {
            return stl::Reduce::norm2(Xx->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.norm2(Xx->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

f32 ReductionOps::dot(const TensorStorage *Xx, const TensorStorage *Xy, const size_t N) const {
    if (this->d_type == deviceType::host) {
        if (N > this->max_dim) {
            return avx2::reduce::dot(Xx->data(), Xy->data(), N);
        } else {
            return stl::Reduce::dot(Xx->data(), Xy->data(), N);
        }
    } else {
        #if CXM_IS_CUDA_AVAILABLE
            cuda::ReduceOp reduce_op;
            return reduce_op.dot(Xx->data(), Xy->data(), N);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}
