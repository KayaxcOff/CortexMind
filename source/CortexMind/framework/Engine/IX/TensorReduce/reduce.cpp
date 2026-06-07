//
// Created by muham on 5.06.2026.
//

#include "CortexMind/framework/Engine/IX/TensorReduce/reduce.hpp"
#include <CortexMind/framework/Engine/AVX2/reduce.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/reduce.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void TensorReduce::sum(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::sum(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ReduceOp::sum(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::sum(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::sum(const TensorStorage *Xx, TensorStorage *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::sum(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::sum_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::sum(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::mean(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::mean(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ReduceOp::mean(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::mean(Xx->data(), Xz->data(), Xz->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::mean(const TensorStorage *Xx, TensorStorage *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::mean(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::mean_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::mean(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::var(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::var(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ReduceOp::var(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::var(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::var(const TensorStorage *Xx, TensorStorage *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::var(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            //cuda::ReduceOp::var_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::var(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::stdv(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::stdv(Xx->data(), Xz->data(), Xx->size());
        } else {
            cuda::ReduceOp::stdv(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::stdv(Xx->data(), Xz->data(), Xz->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::stdv(const TensorStorage *Xx, TensorStorage *Xz, size_t outer_size, size_t dim_size, size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::stdv(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            //cuda::ReduceOp::stdv_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::stdv(Xx->data(), Xz->data(), Xz->size(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::min(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::min(Xx->data(), Xz->data(), Xx->size());
        } else {
            //cuda::ReduceOp::min(Xx->data(), Xz->data(), Xz->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::stdv(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::min(const TensorStorage *Xx, TensorStorage *Xz, size_t outer_size, size_t dim_size, size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));
    CXM_ASSERT(Xx->size() != Xz->size(), "Size of input Storage and output Storage must be same");

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::min(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::min_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::min(Xx->data(), Xz->data(), Xz->size(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::max(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::max(Xx->data(), Xz->data(), Xx->size());
        } else {
            //cuda::ReduceOp::max(Xx->data(), Xz->data(), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::max(Xx->data(), Xz->data(), Xx->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::max(const TensorStorage *Xx, TensorStorage *Xz, size_t outer_size, size_t dim_size, size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::max(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::max_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::max(Xx->data(), Xz->data(), Xz->size(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::norm1(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::norm1(Xx->data(), Xz->data(), Xx->size());
        } else {
            //cuda::ReduceOp::norm1(Xx->data(), Xz->data(), Xz->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::norm1(Xx->data(), Xz->data(), Xz->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::norm1(const TensorStorage *Xx, TensorStorage *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::norm1(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::norm1_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::norm1(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::norm2(const TensorStorage *Xx, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::norm2(Xx->data(), Xz->data(), Xx->size());
        } else {
            //cuda::ReduceOp::norm2(Xx->data(), Xz->data(), Xz->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::norm2(Xx->data(), Xz->data(), Xz->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::norm2(const TensorStorage *Xx, TensorStorage *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xz->device(), "Input Storage's device is " + as_string(Xx->device()) + " and output Storage's device is " + as_string(Xz->device()));

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::norm2(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::norm2_dim(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::norm2(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::argmax(const TensorStorage *Xx, i64 *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Index pointer is null");

    #if CXM_IS_CUDA_AVAILABLE
        if (Xx->device() == DeviceType::kHOST) {
            avx2::reduce::argmax(Xx->data(), Xz, Xx->size());
        } else {
            cuda::ReduceOp::argmax(Xx->data(), reinterpret_cast<i32*>(Xz), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::norm2(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::argmax(const TensorStorage *Xx, i64 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Index pointer is null");

    #if CXM_IS_CUDA_AVAILABLE
        if (Xx->device() == DeviceType::kHOST) {
            avx2::reduce::argmax(Xx->data(), Xz, outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::argmax_dim(Xx->data(), reinterpret_cast<i32*>(Xz), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::argmax(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::argmin(const TensorStorage *Xx, i64 *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Index pointer is null");

    #if CXM_IS_CUDA_AVAILABLE
        if (Xx->device() == DeviceType::kHOST) {
            avx2::reduce::argmin(Xx->data(), Xz, Xx->size());
        } else {
            cuda::ReduceOp::argmin(Xx->data(), reinterpret_cast<i32*>(Xz), Xx->size());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::argmin(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::argmin(const TensorStorage *Xx, i64 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Index pointer is null");

    #if CXM_IS_CUDA_AVAILABLE
        if (Xx->device() == DeviceType::kHOST) {
            avx2::reduce::argmin(Xx->data(), Xz, outer_size, dim_size, inner_size);
        } else {
            cuda::ReduceOp::argmin_dim(Xx->data(), reinterpret_cast<i32*>(Xz), outer_size, dim_size, inner_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::argmin(Xx->data(), Xz->data(), outer_size, dim_size, inner_size);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorReduce::dot(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");

    CXM_ASSERT(Xz == nullptr, "Output Storage is null");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    CXM_ASSERT(Xx->device() != Xy->device() && Xy->device() != Xz->device()," Input Storage's device is " + as_string(Xx->device()) + " and other input Storage's device is " + as_string(Xy->device()) + " and output Storage's device is " + as_string(Xz->device()) );

    const auto dev = Xz->device();

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::reduce::dot(Xx->data(), Xy->data(), Xz->data(), Xz->size());
        } else {

        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::reduce::dot(Xx->data(), Xy->data(), Xz->data(), Xz->size());
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}
