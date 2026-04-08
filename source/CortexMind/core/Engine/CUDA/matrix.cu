//
// Created by muham on 8.04.2026.
//

#include "CortexMind/core/Engine/CUDA/matrix.h"
#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Engine/CUDA/Kernels/matrix.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/operations.h>

using namespace cortex::_fw::cuda;

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