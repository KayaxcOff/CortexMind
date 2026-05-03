//
// Created by muham on 8.04.2026.
//

#include "CortexMind/core/Engine/CUDA/matrix.h"
#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Engine/CUDA/Kernels/matrix.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/operations.h>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/runtime/provider.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void Matrix::add(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::matrix<ops::Addition><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, Xz4, N);
}

void Matrix::sub(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::matrix<ops::Subtraction><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, Xz4, N);
}

void Matrix::mul(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::matrix<ops::Multiplication><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, Xz4, N);
}

void Matrix::div(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::matrix<ops::Division><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, Xz4, N);
}

void Matrix::matmul(const f32* Xx, const f32* Xy, f32* Xz, const size_t xN, const size_t yN, const size_t zN) {
    const f32 alpha = 1.0f;
    const f32 beta = 0.0f;

    CXM_ASSERT(
        cublasSgemm(
            runtime::Provider::instance().handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<i32>(zN),
            static_cast<i32>(xN),
            static_cast<i32>(yN),
            &alpha,
            Xy, static_cast<i32>(zN),
            Xx, static_cast<i32>(yN),
            &beta,
            Xz, static_cast<i32>(zN)
        ) == CUBLAS_STATUS_SUCCESS,
        "cortex::_fw::cuda::Matrix::matmul()",
        "cublasSgemm() failed"
    );
}

void Matrix::add(f32* Xx, const f32* Xy, const size_t N) {
    f32x4*       Xx4 = reinterpret_cast<f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    kernels::matrix_inplace<ops::Addition><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, N);
}

void Matrix::sub(f32* Xx, const f32* Xy, const size_t N) {
    f32x4*       Xx4 = reinterpret_cast<f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    kernels::matrix_inplace<ops::Subtraction><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, N);
}

void Matrix::mul(f32* Xx, const f32* Xy, const size_t N) {
    f32x4*       Xx4 = reinterpret_cast<f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    kernels::matrix_inplace<ops::Multiplication><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, N);
}

void Matrix::div(f32* Xx, const f32* Xy, const size_t N) {
    f32x4*       Xx4 = reinterpret_cast<f32x4*>(Xx);
    const f32x4* Xy4 = reinterpret_cast<const f32x4*>(Xy);
    kernels::matrix_inplace<ops::Division><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xy4, N);
}