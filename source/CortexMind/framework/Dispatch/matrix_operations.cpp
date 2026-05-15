//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Dispatch/matrix_operations.hpp"
#include <CortexMind/framework/Engine/AVX2/matrix.hpp>
#include <CortexMind/framework/Engine/AVX2/broadcast.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/matrix.cuh>
    #include <CortexMind/framework/Engine/CUDA/broadcast.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw::disp;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

Matrix::Matrix(const DeviceType _d_type) : d_type(_d_type) {}

Matrix::~Matrix() = default;

void Matrix::SetDevice(const DeviceType _d_type) {
    this->d_type = _d_type;
}

void Matrix::add( const TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x, const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y, TensorStorage* Xz, const std::vector<i64>& shape_z, const std::vector<i64>& stride_z) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr || Xz == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    dispatch_add(Xx->data(), Xy->data(), Xz->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::sub(const TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, TensorStorage *Xz, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr || Xz == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    dispatch_sub(Xx->data(), Xy->data(), Xz->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::mul(const TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, TensorStorage *Xz, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr || Xz == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    dispatch_mul(Xx->data(), Xy->data(), Xz->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::div(const TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y, TensorStorage *Xz, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr || Xz == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    dispatch_div(Xx->data(), Xy->data(), Xz->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::add(TensorStorage* Xx, const std::vector<i64>& shape_x, const std::vector<i64>& stride_x, const TensorStorage* Xy, const std::vector<i64>& shape_y, const std::vector<i64>& stride_y) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    const auto stride_z = compute_stride(shape_z);

    dispatch_add(Xx->data(), Xy->data(), Xx->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::sub(TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    const auto stride_z = compute_stride(shape_z);

    dispatch_sub(Xx->data(), Xy->data(), Xx->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::mul(TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    const auto stride_z = compute_stride(shape_z);

    dispatch_mul(Xx->data(), Xy->data(), Xx->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::div(TensorStorage *Xx, const std::vector<i64> &shape_x, const std::vector<i64> &stride_x, const TensorStorage *Xy, const std::vector<i64> &shape_y, const std::vector<i64> &stride_y) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr, "Storage is null");
    CXM_ASSERT(!is_broadcastable(shape_x, shape_y), "Shapes are not broadcastable");

    const BroadcastKind kind = classify_broadcast(shape_x, shape_y);
    const auto shape_z  = broadcast_shape(shape_x, shape_y);
    const auto stride_z = compute_stride(shape_z);

    dispatch_div(Xx->data(), Xy->data(), Xx->data(), stride_x, stride_y, shape_x, shape_y, shape_z, stride_z, kind);
}

void Matrix::matmul(const TensorStorage *Xx, const TensorStorage *Xy, TensorStorage *Xz, const size_t xN, const size_t yN, const size_t zN) const {
    CXM_ASSERT(Xx == nullptr || Xy == nullptr || Xz == nullptr, "Storage is null");
    CXM_ASSERT(xN == 0 || yN == 0 || zN == 0, "Matrix dimensions must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (this->d_type == DeviceType::kHOST) {
            avx2::matrix_t::matmul(Xx->data(), Xy->data(), Xz->data(), xN, yN, zN);
        } else {
            cuda::Matrix::matmul(Xx->data(), Xy->data(), Xz->data(), xN, yN, zN);
        }
    #else
        avx2::matrix_t::matmul(Xx->data(), Xy->data(), Xz->data(), xN, yN, zN);
    #endif
}

void Matrix::dispatch_add(const f32 *x, const f32 *y, f32 *z, const std::vector<i64> &sx, const std::vector<i64> &sy, const std::vector<i64> &shape_x, const std::vector<i64> &shape_y, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z, BroadcastKind kind) const {
    const size_t N  = compute_size(shape_z);
    const size_t M  = shape_z.size() >= 2 ? static_cast<size_t>(shape_z[shape_z.size()-2]) : 1;
    const auto Nc = static_cast<size_t>(shape_z.back());

    switch (kind) {
        case BroadcastKind::kNone:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::matrix_t::add(x, y, z, N);
                else
                    cuda::Matrix::add(x, y, z, N);
            #else
                avx2::matrix_t::add(x, y, z, N);
            #endif
            break;

        case BroadcastKind::kRow:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::row_add(x, y, z, M, Nc);
                else
                    cuda::Broadcast::row_add(x, y, z, M, Nc);
            #else
                avx2::Broadcast::row_add(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kCol:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::col_add(x, y, z, M, Nc);
                else
                    cuda::Broadcast::col_add(x, y, z, M, Nc);
            #else
                avx2::Broadcast::col_add(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kGeneral: {
            const auto info = make_broadcast_info(
                shape_x, stride_x, shape_y, stride_y, shape_z, stride_z);
            for (size_t i = 0; i < N; ++i) {
                size_t ox = 0, oy = 0, oz = 0, idx = i;
                for (i32 d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = idx % info.shape[d];
                    ox += coord * info.stride_x[d];
                    oy += coord * info.stride_y[d];
                    oz += coord * info.stride_z[d];
                    idx /= info.shape[d];
                }
                z[oz] = x[ox] + y[oy];
            }
            break;
        }
    }
}

void Matrix::dispatch_sub(const f32 *x, const f32 *y, f32 *z, const std::vector<i64> &sx, const std::vector<i64> &sy, const std::vector<i64> &shape_x, const std::vector<i64> &shape_y, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z, BroadcastKind kind) const {
    const size_t N  = compute_size(shape_z);
    const size_t M  = shape_z.size() >= 2 ? static_cast<size_t>(shape_z[shape_z.size()-2]) : 1;
    const auto Nc = static_cast<size_t>(shape_z.back());

    switch (kind) {
        case BroadcastKind::kNone:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::matrix_t::sub(x, y, z, N);
                else
                    cuda::Matrix::sub(x, y, z, N);
            #else
                avx2::matrix_t::sub(x, y, z, N);
            #endif
            break;

        case BroadcastKind::kRow:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::row_sub(x, y, z, M, Nc);
                else
                    cuda::Broadcast::row_sub(x, y, z, M, Nc);
            #else
                avx2::Broadcast::row_sub(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kCol:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::col_sub(x, y, z, M, Nc);
                else
                    cuda::Broadcast::col_sub(x, y, z, M, Nc);
            #else
                avx2::Broadcast::col_sub(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kGeneral: {
            const auto info = make_broadcast_info(
                shape_x, stride_x, shape_y, stride_y, shape_z, stride_z);
            // General: scalar loop, her iki backend için aynı
            for (size_t i = 0; i < N; ++i) {
                size_t ox = 0, oy = 0, oz = 0, idx = i;
                for (i32 d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = idx % info.shape[d];
                    ox += coord * info.stride_x[d];
                    oy += coord * info.stride_y[d];
                    oz += coord * info.stride_z[d];
                    idx /= info.shape[d];
                }
                z[oz] = x[ox] - y[oy];
            }
            break;
        }
    }
}

void Matrix::dispatch_mul(const f32 *x, const f32 *y, f32 *z, const std::vector<i64> &sx, const std::vector<i64> &sy, const std::vector<i64> &shape_x, const std::vector<i64> &shape_y, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z, BroadcastKind kind) const {
    const size_t N  = compute_size(shape_z);
    const size_t M  = shape_z.size() >= 2 ? static_cast<size_t>(shape_z[shape_z.size()-2]) : 1;
    const size_t Nc = static_cast<size_t>(shape_z.back());

    switch (kind) {
        case BroadcastKind::kNone:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::matrix_t::sub(x, y, z, N);
                else
                    cuda::Matrix::sub(x, y, z, N);
            #else
                avx2::matrix_t::sub(x, y, z, N);
            #endif
            break;

        case BroadcastKind::kRow:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::row_sub(x, y, z, M, Nc);
                else
                    cuda::Broadcast::row_sub(x, y, z, M, Nc);
            #else
                avx2::Broadcast::row_sub(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kCol:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::col_mul(x, y, z, M, Nc);
                else
                    cuda::Broadcast::col_mul(x, y, z, M, Nc);
            #else
                avx2::Broadcast::col_mul(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kGeneral: {
            const auto info = make_broadcast_info(
                shape_x, stride_x, shape_y, stride_y, shape_z, stride_z);
            // General: scalar loop, her iki backend için aynı
            for (size_t i = 0; i < N; ++i) {
                size_t ox = 0, oy = 0, oz = 0, idx = i;
                for (i32 d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = idx % info.shape[d];
                    ox += coord * info.stride_x[d];
                    oy += coord * info.stride_y[d];
                    oz += coord * info.stride_z[d];
                    idx /= info.shape[d];
                }
                z[oz] = x[ox] * y[oy];
            }
            break;
        }
    }
}

void Matrix::dispatch_div(const f32 *x, const f32 *y, f32 *z, const std::vector<i64> &sx, const std::vector<i64> &sy, const std::vector<i64> &shape_x, const std::vector<i64> &shape_y, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z, BroadcastKind kind) const {
    const size_t N  = compute_size(shape_z);
    const size_t M  = shape_z.size() >= 2 ? static_cast<size_t>(shape_z[shape_z.size()-2]) : 1;
    const size_t Nc = static_cast<size_t>(shape_z.back());

    switch (kind) {
        case BroadcastKind::kNone:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::matrix_t::div(x, y, z, N);
                else
                    cuda::Matrix::div(x, y, z, N);
            #else
                avx2::matrix_t::div(x, y, z, N);
            #endif
            break;

        case BroadcastKind::kRow:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::row_div(x, y, z, M, Nc);
                else
                    cuda::Broadcast::row_div(x, y, z, M, Nc);
            #else
                avx2::Broadcast::row_div(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kCol:
            #if CXM_IS_CUDA_AVAILABLE
                if (this->d_type == DeviceType::kHOST)
                    avx2::Broadcast::col_div(x, y, z, M, Nc);
                else
                    cuda::Broadcast::col_div(x, y, z, M, Nc);
            #else
                avx2::Broadcast::col_div(x, y, z, M, Nc);
            #endif
            break;

        case BroadcastKind::kGeneral: {
            const auto info = make_broadcast_info(
                shape_x, stride_x, shape_y, stride_y, shape_z, stride_z);
            for (size_t i = 0; i < N; ++i) {
                size_t ox = 0, oy = 0, oz = 0, idx = i;
                for (i32 d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = idx % info.shape[d];
                    ox += coord * info.stride_x[d];
                    oy += coord * info.stride_y[d];
                    oz += coord * info.stride_z[d];
                    idx /= info.shape[d];
                }
                z[oz] = x[ox] / y[oy];
            }
            break;
        }
    }
}
}
