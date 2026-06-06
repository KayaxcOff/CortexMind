//
// Created by muham on 21.05.2026.
//

#include "CortexMind/framework/Engine/CUDA/compare.cuh"
#include <CortexMind/framework/Engine/CUDA/Kernels/matrix.cuh>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/kernel_operations.hpp>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void CompareTo::greater(const f32 *Xx, const f32 *Yy, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Yy4 = reinterpret_cast<const f32x4*>(Yy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::matrix<ops::Greater><<<grid1d(N >> 2), BLOCK_SIZE_1D>>>(Xx4, Yy4, Xz4, N);
}

void CompareTo::greater_eq(const f32 *Xx, const f32 *Yy, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Yy4 = reinterpret_cast<const f32x4*>(Yy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::matrix<ops::GreaterEqual><<<grid1d(N >> 2), BLOCK_SIZE_1D>>>(Xx4, Yy4, Xz4, N);
}

void CompareTo::less(const f32 *Xx, const f32 *Yy, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Yy4 = reinterpret_cast<const f32x4*>(Yy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::matrix<ops::Less><<<grid1d(N >> 2), BLOCK_SIZE_1D>>>(Xx4, Yy4, Xz4, N);
}

void CompareTo::less_eq(const f32 *Xx, const f32 *Yy, f32 *Xz, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Yy4 = reinterpret_cast<const f32x4*>(Yy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::matrix<ops::LessEqual><<<grid1d(N >> 2), BLOCK_SIZE_1D>>>(Xx4, Yy4, Xz4, N);
}

bool CompareTo::equal(const f32 *Xx, const f32 *Yy, size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    const f32x4* Yy4 = reinterpret_cast<const f32x4*>(Yy);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::matrix<ops::Equal><<<grid1d(N >> 2), BLOCK_SIZE_1D>>>(Xx4, Yy4, Xz4, N);
}