//
// Created by muham on 17.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/matrix.cuh"
#include <CortexMind/framework/Engine/CUDA/cast.cuh>
#include <CortexMind/framework/Engine/CUDA/handle.cuh>
#include <CortexMind/framework/Engine/Kernels/binary.cuh>
#include <CortexMind/framework/Engine/Operations/kernel_ops.cuh>
#include <CortexMind/framework/Tools/Error/errors.hpp>
#include <CortexMind/framework/Tools/grid.cuh>
#include <CortexMind/framework/Type/size.hpp>
#include <tlx/concepts.hpp>

using namespace cortex::_fw::nv;

namespace {
    template<typename T>
    struct BlasTraits;

    template<>
    struct BlasTraits<float> {
        static constexpr cudaDataType_t data_type = CUDA_R_32F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    };

    template<>
    struct BlasTraits<tlx::bfloat16> {
        static constexpr cudaDataType_t data_type = CUDA_R_16BF;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    };

    template<>
    struct BlasTraits<tlx::half> {
        static constexpr cudaDataType_t data_type = CUDA_R_16F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    };
} //unnamed namespace

void Matrix::add(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Binary<ops::Addition><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, Xx.size());
    });
}

void Matrix::sub(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Binary<ops::Subtraction><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, Xx.size());
    });
}

void Matrix::mul(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Binary<ops::Multiplication><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, Xx.size());
    });
}

void Matrix::div(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Binary<ops::Division><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, Xx.size());
    });
}

void Matrix::max(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Binary<ops::Max><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, Xx.size());
    });
}

void Matrix::min(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<const T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));
        const auto Xz4 = nv::convert(reinterpret_cast<T*>(Xz.data()));

        kernels::Binary<ops::Min><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xz4, Xx.size());
    });
}

void Matrix::matmul(const TensorView &Xx, const TensorView &Xy, TensorView &Xz) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int M = static_cast<int>(Xx.size());
        const int K = static_cast<int>(Xy.size());
        const int N = static_cast<int>(Xz.size());

        const handle h;

        constexpr float alpha = 1.0f;
        constexpr float beta  = 0.0f;

        auto xx = reinterpret_cast<const T*>(Xx.data());
        auto xy = reinterpret_cast<const T*>(Xy.data());
        auto xz = reinterpret_cast<T*>(Xz.data());

        const cublasStatus_t status = cublasGemmEx(
            h,

            CUBLAS_OP_N,
            CUBLAS_OP_N,

            N,
            M,
            K,

            &alpha,

            xy,
            BlasTraits<T>::data_type,
            N,

            xx,
            BlasTraits<T>::data_type,
            K,

            &beta,

            xz,
            BlasTraits<T>::data_type,
            N,

            BlasTraits<T>::compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );

        CXM_ASSERT(status != CUBLAS_STATUS_SUCCESS, "matmul on CUDA has failed");
    });
}

void Matrix::add(TensorView &Xx, const TensorView &Xy) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));

        kernels::Binary<ops::Addition><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xx.size());
    });
}

void Matrix::sub(TensorView &Xx, const TensorView &Xy) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));

        kernels::Binary<ops::Subtraction><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xx.size());
    });
}

void Matrix::mul(TensorView &Xx, const TensorView &Xy) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));

        kernels::Binary<ops::Multiplication><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xx.size());
    });
}

void Matrix::div(TensorView &Xx, const TensorView &Xy) {
    dispatch(Xx.dtype(), [&]<tlx::arithmetic_like T>(){
        const int grid_size = grid(Xx.size(), sizeOf(Xx.dtype()));

        const auto Xx4 = nv::convert(reinterpret_cast<T*>(Xx.data()));
        const auto Xy4 = nv::convert(reinterpret_cast<const T*>(Xy.data()));

        kernels::Binary<ops::Division><<<grid_size, kBlockSize>>>(Xx4, Xy4, Xx.size());
    });
}