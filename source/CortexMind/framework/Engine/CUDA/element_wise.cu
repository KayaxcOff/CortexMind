//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Engine/CUDA/element_wise.cuh"
#include <CortexMind/framework/Engine/CUDA/Kernels/activation.cuh>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/kernel_operations.hpp>

using namespace cortex::_fw::cuda;

void ElementWise::pow(const f32* Xx, const f32 exp, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Pow{exp});
}

void ElementWise::sqrt(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Sqrt{});
}

void ElementWise::rsqrt(const f32 *Xx, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Rsqrt{});
}

void ElementWise::square(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Square{});
}

void ElementWise::log(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Log{});
}

void ElementWise::exp(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Exp{});
}

void ElementWise::abs(const f32 *Xx, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Abs{});
}

void ElementWise::sin(const f32 *Xx, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Sin{});
}

void ElementWise::cos(const f32 *Xx, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Cos{});
}

void ElementWise::sign(const f32 *Xx, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Sign{});
}

void ElementWise::neg(const f32 *Xx, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Neg{});
}

void ElementWise::clamp(const f32* Xx, const f32 min_val, const f32 max_val, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Clamp{min_val, max_val});
}