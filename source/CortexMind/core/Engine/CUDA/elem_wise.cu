//
// Created by muham on 19.04.2026.
//

#include "CortexMind/core/Engine/CUDA/elem_wise.h"
#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Engine/CUDA/Kernels/activation.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/operations.h>

using namespace cortex::_fw::cuda;

void ElementWise::pow(const f32* Xx, const f32 exp, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Pow><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Pow{exp});
}

void ElementWise::sqrt(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Sqrt><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Sqrt{});
}

void ElementWise::square(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Square><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Square{});
}

void ElementWise::log(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Log><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Log{});
}

void ElementWise::exp(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Exp><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Exp{});
}