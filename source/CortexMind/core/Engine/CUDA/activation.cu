//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Engine/CUDA/activation.cuh"
#include <CortexMind/core/Engine/CUDA/activation_kernel.cuh>
#include <CortexMind/core/Engine/CUDA/cast.cuh>
#include <CortexMind/core/Engine/CUDA/op.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void activation_t::relu(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::Relu><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::leaky_relu(f32* __restrict__ Xx, const f32 alpha, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::leaky_relu_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), alpha, idx);
    CXM_CUDA_CHECK();
}

void activation_t::sigmoid(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::Sigmoid><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::sigmoid_fast(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::SigmoidFast><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::gelu(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::Gelu><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::gelu_exact(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::GeluExact><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::silu(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::Silu><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::swish(f32* __restrict__ Xx, const f32 beta, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::swish_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), beta, idx);
    CXM_CUDA_CHECK();
}


void activation_t::exp(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::Exp><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::log(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::Log><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}

void activation_t::abs(f32* __restrict__ Xx, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::activation_kernel<op::Abs><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), idx);
    CXM_CUDA_CHECK();
}