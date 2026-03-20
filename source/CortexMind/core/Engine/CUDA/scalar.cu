//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Engine/CUDA/scalar.cuh"
#include <CortexMind/core/Engine/CUDA/cast.cuh>
#include <CortexMind/core/Engine/CUDA/op.cuh>
#include <CortexMind/core/Engine/CUDA/scalar.cuh>
#include <CortexMind/core/Engine/CUDA/scalar_kernel.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void ScalarKernel::add(const f32* Xx, f32 value, f32* Xz, size_t idx) {
    if (idx == 0) return;

    const f4x32* dx = to_vec(Xx);
    f4x32* dz = to_vec(Xz);

    const size_t vec_count = (idx + 3) / 4;

    kernels::scalar_kernel<op::Add><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(dx, value, dz, idx);
    CXM_CUDA_CHECK();
}

void ScalarKernel::sub(const f32* Xx, f32 value, f32* Xz, size_t idx) {
    if (idx == 0) return;

    const f4x32* dx = to_vec(Xx);
    f4x32* dz = to_vec(Xz);

    const size_t vec_count = (idx + 3) / 4;

    kernels::scalar_kernel<op::Sub><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(dx, value, dz, idx);
    CXM_CUDA_CHECK();
}

void ScalarKernel::mul(const f32* Xx, f32 value, f32* Xz, size_t idx) {
    if (idx == 0) return;

    const f4x32* dx = to_vec(Xx);
    f4x32* dz = to_vec(Xz);

    const size_t vec_count = (idx + 3) / 4;

    kernels::scalar_kernel<op::Mul><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(dx, value, dz, idx);
    CXM_CUDA_CHECK();
}

void ScalarKernel::div(const f32* Xx, f32 value, f32* Xz, size_t idx) {
    if (idx == 0) return;

    const f4x32* dx = to_vec(Xx);
    f4x32* dz = to_vec(Xz);

    const size_t vec_count = (idx + 3) / 4;

    kernels::scalar_kernel<op::Div><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(dx, value, dz, idx);
    CXM_CUDA_CHECK();
}