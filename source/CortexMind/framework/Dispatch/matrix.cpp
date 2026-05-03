//
// Created by muham on 3.05.2026.
//

#include "CortexMind/framework/Dispatch/matrix.hpp"
#include <CortexMind/core/Engine/AVX2/matrix.hpp>
#include <CortexMind/core/Engine/AVX2/matrix_broadcast.hpp>
#include <CortexMind/core/Engine/STD/matrix.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/matrix.h>
    #include <CortexMind/core/Engine/CUDA/matrix_broadcast.h>
    #include <CortexMind/core/Engine/CUDA/Kernels/matrix.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/tensor_utils.hpp>
#include <CortexMind/runtime/macros.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw::txl;
using namespace cortex::_fw;

MatrixExecutor::MatrixExecutor(const deviceType d_type) : d_type(d_type) {}

MatrixExecutor::~MatrixExecutor() = default;

void MatrixExecutor::SetDevice(const deviceType _d_type) {
    this->d_type = _d_type;
}

void MatrixExecutor::dispatch_binary(
    const TensorStorage* Xx, const TensorStorage* Xy, TensorStorage* Xz,
    void(*avx2_fn)(const f32*, const f32*, f32*, size_t),
    void(*avx2_broadcast_fn)(const f32*, const f32*, f32*, const BroadcastInfo&),
    void(*stl_fn)(const f32*, const f32*, f32*, size_t),
    void(*stl_broadcast_fn)(const f32*, const f32*, f32*, const BroadcastInfo&)
    #if CXM_IS_CUDA_AVAILABLE
    , void(*cuda_fn)(const f32*, const f32*, f32*, size_t)
    , void(*cuda_broadcast_fn)(const f32*, const f32*, f32*, size_t, const BroadcastInfo&)
    #endif
) const {
    const bool equal = shapes_equal(Xx->shape, Xy->shape);

    if (this->d_type == deviceType::host) {
        if (equal) {
            if (Xx->size() > CXM_THRESHOLD) {
                avx2_fn(Xx->data(), Xy->data(), Xz->data(), Xx->size());
            } else {
                stl_fn(Xx->data(), Xy->data(), Xz->data(), Xx->size());
            }
        } else {
            const BroadcastInfo info = compute_broadcast(Xx->shape, Xy->shape);
            const size_t numel = Xz->size();
            if (numel > CXM_THRESHOLD) {
                avx2_broadcast_fn(Xx->data(), Xy->data(), Xz->data(), info);
            } else {
                stl_broadcast_fn(Xx->data(), Xy->data(), Xz->data(), info);
            }
        }
    }
    #if CXM_IS_CUDA_AVAILABLE
    else if (this->d_type == deviceType::cuda) {
        if (equal) {
            cuda_fn(Xx->data(), Xy->data(), Xz->data(), Xx->size());
        } else {
            const BroadcastInfo info = compute_broadcast(Xx->shape, Xy->shape);
            cuda_broadcast_fn(Xx->data(), Xy->data(), Xz->data(), Xz->size(), info);
        }
    }
    #endif
}

void MatrixExecutor::addition(const TensorStorage* Xx, const TensorStorage* Xy, TensorStorage* Xz) const {
    dispatch_binary(Xx, Xy, Xz,
        avx2::matrix_t::add,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            avx2::MatrixBroadcast::generic_broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a + b; });
        },
        stl::matrix::add,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            stl::matrix::broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a + b; });
        }
        #if CXM_IS_CUDA_AVAILABLE
        , cuda::Matrix::add
        , [](const f32* x, const f32* y, f32* z, const size_t N, const BroadcastInfo& info) {
            cuda::MatrixBroadcast::add(x, y, z, N, info);
        }
        #endif
    );
}

void MatrixExecutor::subtraction(const TensorStorage* Xx, const TensorStorage* Xy, TensorStorage* Xz) const {
    dispatch_binary(Xx, Xy, Xz,
        avx2::matrix_t::sub,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            avx2::MatrixBroadcast::generic_broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a - b; });
        },
        stl::matrix::sub,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            stl::matrix::broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a - b; });
        }
        #if CXM_IS_CUDA_AVAILABLE
        , cuda::Matrix::sub
        , [](const f32* x, const f32* y, f32* z, const size_t N, const BroadcastInfo& info) {
            cuda::MatrixBroadcast::sub(x, y, z, N, info);
        }
        #endif
    );
}

void MatrixExecutor::multiply(const TensorStorage* Xx, const TensorStorage* Xy, TensorStorage* Xz) const {
    dispatch_binary(Xx, Xy, Xz,
        avx2::matrix_t::mul,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            avx2::MatrixBroadcast::generic_broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a * b; });
        },
        stl::matrix::mul,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            stl::matrix::broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a * b; });
        }
        #if CXM_IS_CUDA_AVAILABLE
        , cuda::Matrix::mul
        , [](const f32* x, const f32* y, f32* z, const size_t N, const BroadcastInfo& info) {
            cuda::MatrixBroadcast::mul(x, y, z, N, info);
        }
        #endif
    );
}

void MatrixExecutor::division(const TensorStorage* Xx, const TensorStorage* Xy, TensorStorage* Xz) const {
    dispatch_binary(Xx, Xy, Xz,
        avx2::matrix_t::div,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            avx2::MatrixBroadcast::generic_broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a / b; });
        },
        stl::matrix::div,
        [](const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
            stl::matrix::broadcast(x, y, z, info,
                [](const f32 a, const f32 b) { return a / b; });
        }
        #if CXM_IS_CUDA_AVAILABLE
        , cuda::Matrix::div
        , [](const f32* x, const f32* y, f32* z, const size_t N, const BroadcastInfo& info) {
            cuda::MatrixBroadcast::div(x, y, z, N, info);
        }
        #endif
    );
}

void MatrixExecutor::addition(TensorStorage* Xx, const TensorStorage* Xy) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > CXM_THRESHOLD)
            avx2::matrix_t::add(Xx->data(), Xy->data(), Xx->size());
        else
            stl::matrix::add(Xx->data(), Xy->data(), Xx->size());
    }
    #if CXM_IS_CUDA_AVAILABLE
    else if (this->d_type == deviceType::cuda) {
        cuda::Matrix::add(Xx->data(), Xy->data(), Xx->size());
    }
    #endif
}

void MatrixExecutor::subtraction(TensorStorage* Xx, const TensorStorage* Xy) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > CXM_THRESHOLD)
            avx2::matrix_t::sub(Xx->data(), Xy->data(), Xx->size());
        else
            stl::matrix::sub(Xx->data(), Xy->data(), Xx->size());
    }
    #if CXM_IS_CUDA_AVAILABLE
    else if (this->d_type == deviceType::cuda) {
        cuda::Matrix::sub(Xx->data(), Xy->data(), Xx->size());
    }
    #endif
}

void MatrixExecutor::multiply(TensorStorage* Xx, const TensorStorage* Xy) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > CXM_THRESHOLD)
            avx2::matrix_t::mul(Xx->data(), Xy->data(), Xx->size());
        else
            stl::matrix::mul(Xx->data(), Xy->data(), Xx->size());
    }
    #if CXM_IS_CUDA_AVAILABLE
    else if (this->d_type == deviceType::cuda) {
        cuda::Matrix::mul(Xx->data(), Xy->data(), Xx->size());
    }
    #endif
}

void MatrixExecutor::division(TensorStorage* Xx, const TensorStorage* Xy) const {
    if (this->d_type == deviceType::host) {
        if (Xx->size() > CXM_THRESHOLD)
            avx2::matrix_t::div(Xx->data(), Xy->data(), Xx->size());
        else
            stl::matrix::div(Xx->data(), Xy->data(), Xx->size());
    }
    #if CXM_IS_CUDA_AVAILABLE
    else if (this->d_type == deviceType::cuda) {
        cuda::Matrix::div(Xx->data(), Xy->data(), Xx->size());
    }
    #endif
}

void MatrixExecutor::matmul(const TensorStorage* Xx, const TensorStorage* Xy,
                             TensorStorage* Xz, const i64 M, const i64 K, const i64 N) const {
    if (this->d_type == deviceType::host) {
        avx2::matrix_t::matmul(Xx->data(), Xy->data(), Xz->data(), M, K, N);
    }
    #if CXM_IS_CUDA_AVAILABLE
    else if (this->d_type == deviceType::cuda) {
        cuda::Matrix::matmul(Xx->data(), Xy->data(), Xz->data(), M, K, N);
    }
    #endif
}