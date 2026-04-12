//
// Created by muham on 8.04.2026.
//

#include "CortexMind/core/Engine/CUDA/scalar.h"
#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Engine/CUDA/Kernels/scalar.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/operations.h>

using namespace cortex::_fw::cuda;

void ScalarKernel::add(const f32* Xx, const f32 value, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::scalar<ops::Addition><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, Xz4, N);
}

void ScalarKernel::sub(const f32* Xx, const f32 value, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::scalar<ops::Subtraction><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, Xz4, N);
}

void ScalarKernel::mul(const f32* Xx, const f32 value, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::scalar<ops::Multiplication><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, Xz4, N);
}

void ScalarKernel::div(const f32* Xx, const f32 value, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);

    kernels::scalar<ops::Division><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, Xz4, N);
}

void ScalarKernel::add(f32* Xx, const f32 value, const size_t N) {
    f32x4* Xx4 = reinterpret_cast<f32x4*>(Xx);
    kernels::scalar_inplace<ops::Addition><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, N);
}

void ScalarKernel::sub(f32* Xx, const f32 value, const size_t N) {
    f32x4* Xx4 = reinterpret_cast<f32x4*>(Xx);
    kernels::scalar_inplace<ops::Subtraction><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, N);
}

void ScalarKernel::mul(f32* Xx, const f32 value, const size_t N) {
    f32x4* Xx4 = reinterpret_cast<f32x4*>(Xx);
    kernels::scalar_inplace<ops::Multiplication><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, N);
}

void ScalarKernel::div(f32* Xx, const f32 value, const size_t N) {
    f32x4* Xx4 = reinterpret_cast<f32x4*>(Xx);
    kernels::scalar_inplace<ops::Division><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, value, N);
}