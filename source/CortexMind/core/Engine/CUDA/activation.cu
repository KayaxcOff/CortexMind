//
// Created by muham on 31.03.2026.
//

#include "CortexMind/core/Engine/CUDA/activation.h"
#include <CortexMind/core/Kernels/activation.cuh>
#include <CortexMind/core/Engine/CUDA/reduce.h>
#include <CortexMind/core/Tools/utilities.hpp>
#include <CortexMind/core/Tools/ops.h>
#include <CortexMind/core/Kernels/reduce.cuh>

using namespace cortex::_fw::cuda;

void Activation::relu(const bf16* Xx, bf16* Xz, const size_t N) {
    if (N == 0) return;
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xz, N, ops::ReLU{});
}

void Activation::leaky_relu(const bf16* Xx, bf16* Xz, const size_t N, const f32 alpha) {
    if (N == 0) return;
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xz, N, ops::LeakyReLU{alpha});
}

void Activation::sigmoid(const bf16* Xx, bf16* Xz, const size_t N) {
    if (N == 0) return;
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xz, N, ops::Sigmoid{});
}

void Activation::tanh(const bf16* Xx, bf16* Xz, const size_t N) {
    if (N == 0) return;
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xz, N, ops::Tanh{});
}

void Activation::gelu(const bf16* Xx, bf16* Xz, const size_t N) {
    if (N == 0) return;
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xz, N, ops::GeLU{});
}

void Activation::silu(const bf16* Xx, bf16* Xz, const size_t N) {
    if (N == 0) return;
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xz, N, ops::SiLU{});
}

void Activation::swish(const bf16* Xx, bf16* Xz, const size_t N, const f32 beta) {
    if (N == 0) return;
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xz, N, ops::Swish{beta});
}

void Activation::softmax(const bf16* Xx, bf16* Xz, const size_t N) {
    if (N == 0) return;

    const f32 max_val = Reduce::max(Xx, N);
    f32* d_max = nullptr;
    f32* d_sum = nullptr;
    cudaMalloc(&d_max, sizeof(f32));
    cudaMalloc(&d_sum, sizeof(f32));
    cudaMemcpy(d_max, &max_val, sizeof(f32), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(f32));

    // exp(x - max) → Xz, sum topla
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(
        Xx, Xz, N,
        [d_max] __device__ (f32 x) { return expf(x - *d_max); }
    );
    kernels::reduce_sum_kernel<<<grid1d(N), BLOCK_SIZE_1D>>>(Xz, d_sum, N);

    // normalize
    kernels::softmax_scale<<<grid1d(N), BLOCK_SIZE_1D>>>(Xz, d_max, d_sum, N);

    cudaFree(d_max);
    cudaFree(d_sum);
}

// ─────────────────────────────────────────────
// Inplace
// ─────────────────────────────────────────────
void Activation::relu_   (bf16* Xx, const size_t N) {
    if (N == 0) return;
    kernels::inplace_activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, N, ops::ReLU{});
}

void Activation::sigmoid_(bf16* Xx, const size_t N) {
    if (N == 0) return;
    kernels::inplace_activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, N, ops::Sigmoid{});
}

void Activation::gelu_   (bf16* Xx, const size_t N) {
    if (N == 0) return;
    kernels::inplace_activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, N, ops::GeLU{});
}

void Activation::silu_   (bf16* Xx, const size_t N) {
    if (N == 0) return;
    kernels::inplace_activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, N, ops::SiLU{});
}