//
// Created by muham on 12.05.2026.
//

#include "CortexMind/framework/Dispatch/reduce_operations.hpp"
#include <CortexMind/framework/Engine/AVX2/reduce.hpp>
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <limits>

using namespace cortex::_fw::disp;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

Reduce::Reduce(const DeviceType _d_type) : d_type(_d_type) {}

Reduce::~Reduce() = default;

void Reduce::SetDevice(const DeviceType _d_type) {
    CXM_WARN(this->d_type == _d_type, "You already using " + as_string(this->d_type) + " device");
    this->d_type = _d_type;
}

f32 Reduce::sum(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::sum(x->data(), N);
        } else {
            output = this->op.sum(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::sum(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::mean(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::mean(x->data(), N);
        } else {
            output = this->op.mean(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::mean(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::var(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::var(x->data(), N);
        } else {
            output = this->op.var(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::var(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::std(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::std(x->data(), N);
        } else {
            output = this->op.stdv(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::std(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::min(const TensorStorage *x, const size_t N) const {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::min(x->data(), N);
        } else {
            output = this->op.min(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::min(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::max(const TensorStorage *x, const size_t N) const {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::max(x->data(), N);
        } else {
            output = this->op.max(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::max(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::norm1(const TensorStorage *x, const size_t N) const {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::norm1(x->data(), N);
        } else {
            output = this->op.norm1(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::norm1(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::norm2(const TensorStorage *x, const size_t N) const {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::norm2(x->data(), N);
        } else {
            output = this->op.norm2(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::norm2(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}

f32 Reduce::dot(const TensorStorage *Xx, const TensorStorage *Xy, const size_t N) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr, "Storages/Stroage are/is null");
    CXM_ASSERT(N == 0, "Number of tensor must be non-zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::HOST) {
            output = avx2::reduce::dot(Xx->data(), Xy->data(), N);
        } else {
            output = this->op.dot(Xx->data(), Xy->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::dot(Xx->data(), Xy->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE

    return output;
}
