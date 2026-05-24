//
// Created by muham on 16.05.2026.
//

#include "CortexMind/framework/Engine/IX/matrix.hpp"
#include <CortexMind/framework/Engine/AVX2/broadcast.hpp>
#include <CortexMind/framework/Engine/AVX2/broadcast_general.hpp>
#include <CortexMind/framework/Engine/AVX2/matrix.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/broadcast.cuh>
    #include <CortexMind/framework/Engine/CUDA/matrix.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void MatrixOp::dispatch(const f32 *x, const f32 *y, f32 *z, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z, const BroadcastKind kind, const DeviceType dev, const char op) {
    const size_t N  = compute_size(shape_z);
    const size_t M  = shape_z.size() >= 2 ? static_cast<size_t>(shape_z[shape_z.size() - 2]) : 1;
    const auto Nc = static_cast<size_t>(shape_z.back());

    #define CXM_DISPATCH_BACKEND(avx_fn, cuda_fn)   \
        do {                                        \
            if (dev == DeviceType::kHOST) {         \
                avx_fn;                             \
            } else {                                \
                CXM_IS_CUDA_AVAILABLE_GUARD         \
                cuda_fn;                            \
            }                                       \
        } while(0)

    switch (kind) {
        case BroadcastKind::kNone:
            if (op == '+') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::matrix_t::add(x, y, z, N);
                    } else {
                        cuda::Matrix::add(x, y, z, N);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::matrix_t::add(x, y, z, N);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else if (op == '-') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::matrix_t::sub(x, y, z, N);
                    } else {
                        cuda::Matrix::sub(x, y, z, N);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::matrix_t::sub(x, y, z, N);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else if (op == '*') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::matrix_t::mul(x, y, z, N);
                    } else {
                        cuda::Matrix::mul(x, y, z, N);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::matrix_t::mul(x, y, z, N);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::matrix_t::div(x, y, z, N);
                    } else {
                        cuda::Matrix::div(x, y, z, N);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::matrix_t::div(x, y, z, N);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            }
            break;
        case BroadcastKind::kRow:
            if (op == '+') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::row_add(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::row_add(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::row_add(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else if (op == '-') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::row_sub(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::row_sub(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::row_sub(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else if (op == '*') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::row_mul(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::row_mul(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::row_mul(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::row_div(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::row_div(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::row_div(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            }
            break;
        case BroadcastKind::kCol:
            if (op == '+') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::col_add(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::col_add(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::col_add(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else if (op == '-') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::col_sub(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::col_sub(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::col_sub(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else if (op == '*') {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::col_mul(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::col_mul(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::col_mul(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            } else {
                #if CXM_IS_CUDA_AVAILABLE
                    if (dev == DeviceType::kHOST) {
                        avx2::Broadcast::col_div(x, y, z, M, Nc);
                    } else {
                        cuda::Broadcast::col_div(x, y, z, M, Nc);
                    }
                #else //#if CXM_IS_CUDA_AVAILABLE
                    avx2::Broadcast::col_div(x, y, z, M, Nc);
                #endif //#if CXM_IS_CUDA_AVAILABLE #else
            }
            break;
        case BroadcastKind::kGeneral:
            const BroadcastInfo info = make_broadcast_info(shape_x, stride_x, shape_y, stride_y, shape_z, stride_z);
            const size_t total = compute_size(shape_z);

            #if CXM_IS_CUDA_AVAILABLE
                if (dev == DeviceType::kHOST) {
                    if (op == '+') {
                        avx2::general_broadcast_add(x, y, z, info);
                    } else if (op == '-') {
                        avx2::general_broadcast_sub(x, y, z, info);
                    } else if (op == '*') {
                        avx2::general_broadcast_mul(x, y, z, info);
                    } else {
                        avx2::general_broadcast_div(x, y, z, info);
                    }
                } else {
                    if (op == '+') {
                        cuda::Broadcast::general_add(x, y, z, info, total);
                    } else if (op == '-') {
                        cuda::Broadcast::general_sub(x, y, z, info, total);
                    } else if (op == '*') {
                        cuda::Broadcast::general_mul(x, y, z, info, total);
                    } else {
                        cuda::Broadcast::general_div(x, y, z, info, total);
                    }
                }
            #else //#if CXM_IS_CUDA_AVAILABLE
                if (op == '+') {
                    avx2::general_broadcast_add(x, y, z, info);
                } else if (op == '-') {
                    avx2::general_broadcast_sub(x, y, z, info);
                } else if (op == '*') {
                    avx2::general_broadcast_mul(x, y, z, info);
                } else {
                    avx2::general_broadcast_div(x, y, z, info);
                }
            #endif //#if CXM_IS_CUDA_AVAILABLE
            break;
    }
}

void MatrixOp::add(const TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, TensorStorage *Xz, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid() || !Xz->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(), "Storage is empty");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    CXM_ASSERT(Xx->device() != Xz->device(), "Device mismatch: Xx=" + as_string(Xx->device()) + " Xz=" + as_string(Xz->device()));

    dispatch(Xx->data(), Xy->data(), Xz->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '+');
}

void MatrixOp::sub(const TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, TensorStorage *Xz, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid() || !Xz->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(), "Storage is empty");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    CXM_ASSERT(Xx->device() != Xz->device(), "Device mismatch: Xx=" + as_string(Xx->device()) + " Xz=" + as_string(Xz->device()));

    dispatch(Xx->data(), Xy->data(), Xz->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '-');
}

void MatrixOp::mul(const TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, TensorStorage *Xz, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid() || !Xz->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(), "Storage is empty");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    CXM_ASSERT(Xx->device() != Xz->device(), "Device mismatch: Xx=" + as_string(Xx->device()) + " Xz=" + as_string(Xz->device()));

    dispatch(Xx->data(), Xy->data(), Xz->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '*');
}

void MatrixOp::div(const TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, TensorStorage *Xz, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid() || !Xz->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(), "Storage is empty");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    CXM_ASSERT(Xx->device() != Xz->device(), "Device mismatch: Xx=" + as_string(Xx->device()) + " Xz=" + as_string(Xz->device()));

    dispatch(Xx->data(), Xy->data(), Xz->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '/');
}

void MatrixOp::add(TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(),   "Storage is empty");
    //CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    //const auto shape_z  = broadcast_shape(shape_x, shape_y);

    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    CXM_ASSERT(shape_z == shape_x, "In-place operation: broadcasted shape must equal input shape");

    const auto stride_z = compute_stride(shape_z);

    dispatch(Xx->data(), Xy->data(), Xx->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '+');
}

void MatrixOp::sub(TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(),   "Storage is empty");
    //CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    //const auto shape_z  = broadcast_shape(shape_x, shape_y);

    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    CXM_ASSERT(shape_z == shape_x, "In-place operation: broadcasted shape must equal input shape");

    const auto stride_z = compute_stride(shape_z);

    dispatch(Xx->data(), Xy->data(), Xx->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '-');
}

void MatrixOp::mul(TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(),   "Storage is empty");
    //CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    //const auto shape_z  = broadcast_shape(shape_x, shape_y);

    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    CXM_ASSERT(shape_z == shape_x, "In-place operation: broadcasted shape must equal input shape");

    const auto stride_z = compute_stride(shape_z);

    dispatch(Xx->data(), Xy->data(), Xx->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '*');
}

void MatrixOp::div(TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid(), "Storage is null");
    CXM_ASSERT(Xx->isEmpty() || Xy->isEmpty(),   "Storage is empty");
    //CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");
    //const auto shape_z  = broadcast_shape(shape_x, shape_y);

    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    CXM_ASSERT(shape_z == shape_x, "In-place operation: broadcasted shape must equal input shape");

    const auto stride_z = compute_stride(shape_z);

    dispatch(Xx->data(), Xy->data(), Xx->data(), shape_x, stride_x, shape_y, stride_y, shape_z, stride_z, classify_broadcast(shape_x, shape_y), Xx->device(), '/');
}

void MatrixOp::matmul(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz, const size_t xN, const size_t yN, const size_t zN, const DeviceType dev) {
    CXM_ASSERT(!Xx->isValid() || !Xy->isValid() || !Xz->isValid(), "Storage is null");
    CXM_ASSERT(xN == 0 || yN == 0 || zN == 0, "Matrix dimensions must be non-zero");
    CXM_ASSERT(Xx->device() != Xy->device(), "Device mismatch: Xx=" + as_string(Xx->device()) + " Xy=" + as_string(Xy->device()));

    #if CXM_IS_CUDA_AVAILABLE
        if (dev == DeviceType::kHOST) {
            avx2::matrix_t::matmul(Xx->data(), Xy->data(), Xz->data(), xN, yN, zN);
        } else {
            cuda::Matrix::matmul(Xx->data(), Xy->data(), Xz->data(), xN, yN, zN);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        avx2::matrix_t::matmul(Xx->data(), Xy->data(), Xz->data(), xN, yN, zN);
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}