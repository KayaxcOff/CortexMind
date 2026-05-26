//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Engine/IX/reduce.hpp"
#include <CortexMind/framework/Engine/AVX2/reduce.hpp>
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>
#include <limits>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

reduce::reduce() = default;

reduce::~reduce() = default;

#if CXM_IS_CUDA_AVAILABLE
    static cuda::ReduceOp op;
#endif //#if CXM_IS_CUDA_AVAILABLE

f32 reduce::sum(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::sum(x->data(), N);
        } else {
            output = op.sum(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::sum(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::mean(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::mean(x->data(), N);
        } else {
            output = op.mean(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::mean(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::var(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::var(x->data(), N);
        } else {
            output = op.var(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::var(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::stdv(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::std(x->data(), N);
        } else {
            output = op.stdv(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::std(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::min(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::min(x->data(), N);
        } else {
            output = op.min(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::min(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::max(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::max(x->data(), N);
        } else {
            output = op.max(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::max(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::norm1(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::norm1(x->data(), N);
        } else {
            output = op.norm1(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::norm1(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::norm2(const TensorStorage *x, const size_t N) {
    CXM_ASSERT(!x->isValid(), "Input Storage is null");
    CXM_ASSERT(x->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    const auto dev = x->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::norm2(x->data(), N);
        } else {
            output = op.norm2(x->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::norm2(x->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

f32 reduce::dot(const TensorStorage *Xx, const TensorStorage *Xy, const size_t N) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(N <= 0, "Number element of tensor must be higher than zero");

    CXM_ASSERT(!Xy->isValid(), "Input Storage is null");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xy->device()));

    f32 output = std::numeric_limits<f32>::quiet_NaN();

    auto dev = DeviceType::kHOST;

    if (Xx->device() == Xy->device()) {
        dev = Xy->device();
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            output = avx2::reduce::dot(Xx->data(), Xy->data(), N);
        } else {
            output = op.dot(Xx->data(), Xy->data(), N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        output = avx2::reduce::dot(Xx->data(), Xy->data(), N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}

void reduce::sum(const TensorStorage *Xx, TensorStorage *Xz, const std::vector<i64> &shape, const std::vector<i64> &dims, const std::vector<i64> &out_shape) {
    CXM_ASSERT(!Xx->isValid(), "Input Storage is null");
    CXM_ASSERT(Xx->isEmpty(),  "Input Storage is empty");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is null");
    CXM_ASSERT(Xz->isEmpty(),  "Output Storage is empty");
    CXM_ASSERT(Xx->device() != Xz->device(), "Device mismatch");

    const DeviceType device  = Xx->device();
    const size_t     ndim    = shape.size();
    const size_t     total_in  = compute_size(shape);
    const size_t     total_out = compute_size(out_shape);

    std::memset(Xz->data(), 0, total_out * sizeof(f32));

    const bool last_dim_only  = (dims.size() == 1 && dims[0] == static_cast<i64>(ndim - 1));
    const bool first_dim_only = (dims.size() == 1 && dims[0] == 0);

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kCUDA) {
            if (last_dim_only) {
                const auto cols = static_cast<size_t>(shape.back());
                const size_t rows = total_in / cols;
                cuda::ReduceOp::sum_last_dim(Xx->data(), Xz->data(), rows, cols);
            } else if (first_dim_only) {
                const auto rows = static_cast<size_t>(shape[0]);
                const size_t cols = total_in / rows;
                cuda::ReduceOp::sum_first_dim(Xx->data(), Xz->data(), rows, cols);
            } else {
                CXM_ASSERT(true, "sum_dim CUDA: mixed dims not implemented");
            }
            return;
        } else if (device == DeviceType::kHOST) {

            if (last_dim_only) {
                const auto cols = static_cast<size_t>(shape.back());
                const size_t rows = total_in / cols;
                avx2::reduce::sum_last_dim(Xx->data(), Xz->data(), rows, cols);
                return;
            }

            if (first_dim_only) {
                const auto rows = static_cast<size_t>(shape[0]);
                const size_t cols = total_in / rows;
                avx2::reduce::sum_first_dim(Xx->data(), Xz->data(), rows, cols);
                return;
            }

            const auto out_strides = compute_stride(out_shape);
            std::vector is_reduced(ndim, false);
            for (const i64 d : dims) is_reduced[static_cast<size_t>(d)] = true;

            for (size_t i = 0; i < total_in; ++i) {
                size_t oz  = 0;
                size_t idx = i;
                for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
                    const auto ud = static_cast<size_t>(d);
                    const size_t coord = idx % static_cast<size_t>(shape[ud]);
                    idx /= static_cast<size_t>(shape[ud]);
                    if (!is_reduced[ud]) {
                        oz += coord * static_cast<size_t>(out_strides[ud]);
                    }
                }
                Xz->data()[oz] += Xx->data()[i];
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE

        if (last_dim_only) {
            const auto cols = static_cast<size_t>(shape.back());
            const size_t rows = total_in / cols;
            avx2::reduce::sum_last_dim(Xx->data(), Xz->data(), rows, cols);
            return;
        }

        if (first_dim_only) {
            const auto rows = static_cast<size_t>(shape[0]);
            const size_t cols = total_in / rows;
            avx2::reduce::sum_first_dim(Xx->data(), Xz->data(), rows, cols);
            return;
        }

        const auto out_strides = compute_stride(out_shape);
        std::vector<bool> is_reduced(ndim, false);
        for (const i64 d : dims) is_reduced[static_cast<size_t>(d)] = true;

        for (size_t i = 0; i < total_in; ++i) {
            size_t oz  = 0;
            size_t idx = i;
            for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
                const auto ud = static_cast<size_t>(d);
                const size_t coord = idx % static_cast<size_t>(shape[ud]);
                idx /= static_cast<size_t>(shape[ud]);
                if (!is_reduced[ud]) {
                    oz += coord * static_cast<size_t>(out_strides[ud]);
                }
            }
            Xz->data()[oz] += Xx->data()[i];
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else


}