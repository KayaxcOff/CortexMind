//
// Created by muham on 6.06.2026.
//

#include "CortexMind/framework/Engine/IX/TensorOp/op.hpp"
#include <CortexMind/framework/Engine/AVX2/broadcast.hpp>
#include <CortexMind/framework/Engine/AVX2/broadcast_general.hpp>
#include <CortexMind/framework/Engine/AVX2/matrix.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/broadcast.cuh>
    #include <CortexMind/framework/Engine/CUDA/matrix.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void TensorOp::add(const TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y, TensorStorage *Xz, const TensorShape &shape_z) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xz->device();

    const size_t M  = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::add(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                } else {
                    cuda::Matrix::add(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_add(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::row_add(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_add(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::col_add(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_add(Xx->data(), Xy->data(), Xz->data(), info);
                } else {
                    cuda::Broadcast::general_add(Xx->data(), Xy->data(), Xz->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::add(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_add(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_add(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_add(Xx->data(), Xy->data(), Xz->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::sub(const TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y, TensorStorage *Xz, const TensorShape &shape_z) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xz->device();

    const size_t M  = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::sub(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                } else {
                    cuda::Matrix::sub(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_sub(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::row_sub(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_sub(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::col_sub(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_sub(Xx->data(), Xy->data(), Xz->data(), info);
                } else {
                    cuda::Broadcast::general_sub(Xx->data(), Xy->data(), Xz->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::sub(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_sub(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_sub(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_sub(Xx->data(), Xy->data(), Xz->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::mul(const TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y, TensorStorage *Xz, const TensorShape &shape_z) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xz->device();

    const size_t M  = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::mul(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                } else {
                    cuda::Matrix::mul(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_mul(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::row_mul(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_mul(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::col_mul(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_mul(Xx->data(), Xy->data(), Xz->data(), info);
                } else {
                    cuda::Broadcast::general_mul(Xx->data(), Xy->data(), Xz->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::mul(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_mul(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_mul(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_mul(Xx->data(), Xy->data(), Xz->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::div(const TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y, TensorStorage *Xz, const TensorShape &shape_z) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xz->device();

    const size_t M  = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::div(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                } else {
                    cuda::Matrix::div(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_div(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::row_div(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_div(Xx->data(), Xy->data(), Xz->data(), M, N);
                } else {
                    cuda::Broadcast::col_div(Xx->data(), Xy->data(), Xz->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_div(Xx->data(), Xy->data(), Xz->data(), info);
                } else {
                    cuda::Broadcast::general_div(Xx->data(), Xy->data(), Xz->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::div(Xx->data(), Xy->data(), Xz->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_div(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_div(Xx->data(), Xy->data(), Xz->data(), M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_div(Xx->data(), Xy->data(), Xz->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::add(TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y) {
    CXM_ASSERT(Xx == nullptr, "Input/Output Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input/Output Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");


    const TensorShape shape_z = broadcast_shape(shape_x, shape_y);

    CXM_ASSERT(shape_x.ndim != shape_z.ndim, "In-place error: Dimension count mismatch!");
    for (i32 i = 0; i < shape_x.ndim; ++i) {
        CXM_ASSERT(shape_x.shape[i] != shape_z.shape[i], "In-place error: Xx shape cannot be expanded!");
    }

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xx->device();

    const size_t M = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::add(Xx->data(), Xy->data(), Xx->size());
                } else {
                    cuda::Matrix::add(Xx->data(), Xy->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_add(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::row_add(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_add(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::col_add(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_add(Xx->data(), Xy->data(), Xx->data(), info);
                } else {
                    cuda::Broadcast::general_add(Xx->data(), Xy->data(), Xx->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::add(Xx->data(), Xy->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_add(Xx->data(), Xy->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_add(Xx->data(), Xy->data(),  M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_add(Xx->data(), Xy->data(), Xx->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::sub(TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y) {
    CXM_ASSERT(Xx == nullptr, "Input/Output Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input/Output Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");


    const TensorShape shape_z = broadcast_shape(shape_x, shape_y);

    CXM_ASSERT(shape_x.ndim != shape_z.ndim, "In-place error: Dimension count mismatch!");
    for (i32 i = 0; i < shape_x.ndim; ++i) {
        CXM_ASSERT(shape_x.shape[i] != shape_z.shape[i], "In-place error: Xx shape cannot be expanded!");
    }

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xx->device();

    const size_t M = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::sub(Xx->data(), Xy->data(), Xx->size());
                } else {
                    cuda::Matrix::sub(Xx->data(), Xy->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_sub(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::row_sub(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_sub(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::col_sub(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_sub(Xx->data(), Xy->data(), Xx->data(), info);
                } else {
                    cuda::Broadcast::general_sub(Xx->data(), Xy->data(), Xx->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::sub(Xx->data(), Xy->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_sub(Xx->data(), Xy->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_sub(Xx->data(), Xy->data(),  M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_sub(Xx->data(), Xy->data(), Xx->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::mul(TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y) {
        CXM_ASSERT(Xx == nullptr, "Input/Output Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input/Output Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");


    const TensorShape shape_z = broadcast_shape(shape_x, shape_y);

    CXM_ASSERT(shape_x.ndim != shape_z.ndim, "In-place error: Dimension count mismatch!");
    for (i32 i = 0; i < shape_x.ndim; ++i) {
        CXM_ASSERT(shape_x.shape[i] != shape_z.shape[i], "In-place error: Xx shape cannot be expanded!");
    }

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xx->device();

    const size_t M = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::mul(Xx->data(), Xy->data(), Xx->size());
                } else {
                    cuda::Matrix::mul(Xx->data(), Xy->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_mul(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::row_mul(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_mul(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::col_mul(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_mul(Xx->data(), Xy->data(), Xx->data(), info);
                } else {
                    cuda::Broadcast::general_mul(Xx->data(), Xy->data(), Xx->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::mul(Xx->data(), Xy->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_mul(Xx->data(), Xy->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_mul(Xx->data(), Xy->data(),  M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_mul(Xx->data(), Xy->data(), Xx->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::div(TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y) {
    CXM_ASSERT(Xx == nullptr, "Input/Output Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(!Xx->isValid(), "Input/Output Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");


    const TensorShape shape_z = broadcast_shape(shape_x, shape_y);

    CXM_ASSERT(shape_x.ndim != shape_z.ndim, "In-place error: Dimension count mismatch!");
    for (i32 i = 0; i < shape_x.ndim; ++i) {
        CXM_ASSERT(shape_x.shape[i] != shape_z.shape[i], "In-place error: Xx shape cannot be expanded!");
    }

    const auto kind = classify_broadcast(shape_x, shape_y);
    const auto dev = Xx->device();

    const size_t M = shape_z.ndim >= 2 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 2]) : 1;
    const size_t N = shape_z.ndim >= 1 ? static_cast<size_t>(shape_z.shape[shape_z.ndim - 1]) : 1;

    #if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                if (dev == DeviceType::kHOST) {
                    avx2::matrix_t::div(Xx->data(), Xy->data(), Xx->size());
                } else {
                    cuda::Matrix::div(Xx->data(), Xy->data(), Xx->size());
                }
                break;
            }
            case BroadcastKind::kRow: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::row_div(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::row_div(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kCol: {
                if (dev == DeviceType::kHOST) {
                    avx2::Broadcast::col_div(Xx->data(), Xy->data(), M, N);
                } else {
                    cuda::Broadcast::col_div(Xx->data(), Xy->data(), M, N);
                }
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                if (dev == DeviceType::kHOST) {
                    avx2::general_broadcast_div(Xx->data(), Xy->data(), Xx->data(), info);
                } else {
                    cuda::Broadcast::general_div(Xx->data(), Xy->data(), Xx->data(), info, N);
                }
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        switch (kind) {
            case BroadcastKind::kNone: {
                avx2::matrix_t::div(Xx->data(), Xy->data(), Xx->size());
                break;
            }
            case BroadcastKind::kRow: {
                avx2::Broadcast::row_div(Xx->data(), Xy->data(), M, N);
                break;
            }
            case BroadcastKind::kCol: {
                avx2::Broadcast::col_div(Xx->data(), Xy->data(),  M, N);
                break;
            }
            case BroadcastKind::kGeneral: {
                const BroadcastInfo info = make_broadcast_info(shape_x, shape_y, shape_z);
                avx2::general_broadcast_div(Xx->data(), Xy->data(), Xx->data(), info);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorOp::matmul(const TensorStorage *Xx, const TensorShape &shape_x, const TensorStorage *Xy, const TensorShape &shape_y, TensorStorage *Xz, const TensorShape &shape_z) {
    CXM_ASSERT(Xx == nullptr, "Input Storage is null");
    CXM_ASSERT(Xy == nullptr, "Input Storage is null");
    CXM_ASSERT(Xz == nullptr, "Output Storage is null");

    CXM_ASSERT(!Xx->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xy->isValid(), "Input Storage is invalid");
    CXM_ASSERT(!Xz->isValid(), "Output Storage is invalid");

    CXM_ASSERT(Xx->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xy->isEmpty(), "Input Storage is empty");
    CXM_ASSERT(Xz->isEmpty(), "Output Storage is empty");

    const auto dev = Xz->device();

    const auto M = static_cast<size_t>(shape_x.shape[shape_x.ndim - 2]);
    const auto K = static_cast<size_t>(shape_x.shape[shape_x.ndim - 1]);
    const auto N = static_cast<size_t>(shape_y.shape[shape_y.ndim - 1]);

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::matrix_t::matmul(Xx->data(), Xy->data(), Xz->data(), M, K, N);
        } else {
            cuda::Matrix::matmul(Xx->data(), Xy->data(), Xz->data(), M, K, N);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::matrix_t::matmul(Xx->data(), Xy->data(), Xz->data(), M, K, N);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}
