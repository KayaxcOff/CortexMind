//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Engine/CUDA/inplace.cuh"
#include <CortexMind/core/Engine/CUDA/cast.cuh>
#include <CortexMind/core/Engine/CUDA/inplace_kernel.cuh>
#include <CortexMind/core/Engine/CUDA/op.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void inplace::add(f32* __restrict Xx, const f32* __restrict Xy, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_kernel<op::Add><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), idx);
    CXM_CUDA_CHECK();
}

void inplace::sub(f32* __restrict Xx, const f32* __restrict Xy, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_kernel<op::Sub><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), idx);
    CXM_CUDA_CHECK();
}

void inplace::mul(f32* __restrict Xx, const f32* __restrict Xy, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_kernel<op::Mul><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), idx);
    CXM_CUDA_CHECK();
}

void inplace::div(f32* __restrict Xx, const f32* __restrict Xy, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_kernel<op::Div><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), idx);
    CXM_CUDA_CHECK();
}

void inplace_scalar::add(f32* __restrict Xx, const f32 value, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_scalar_kernel<op::Add><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), value, idx);
    CXM_CUDA_CHECK();
}

void inplace_scalar::sub(f32* __restrict Xx, const f32 value, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_scalar_kernel<op::Sub><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), value, idx);
    CXM_CUDA_CHECK();
}

void inplace_scalar::mul(f32* __restrict Xx, const f32 value, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_scalar_kernel<op::Mul><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), value, idx);
    CXM_CUDA_CHECK();
}

void inplace_scalar::div(f32* __restrict Xx, const f32 value, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::inplace_scalar_kernel<op::Div><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), value, idx);
    CXM_CUDA_CHECK();
}