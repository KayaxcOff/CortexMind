//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Engine/CUDA/activation.cuh"
#include <CortexMind/framework/Engine/CUDA/Kernels/activation.cuh>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/kernel_operations.hpp>

using namespace cortex::_fw::cuda;

void Activation::relu(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::ReLU{});
}

void Activation::leaky_relu(const f32* Xx, f32* Xz, const size_t N, const f32 alpha) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::LeakyReLU{alpha});
}

void Activation::sigmoid(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Sigmoid{});
}

void Activation::sigmoid_fast(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SigmoidFast{});
}

void Activation::tanh(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Tanh{});
}

void Activation::gelu(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::GELU{});
}

void Activation::gelu_exact(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::GELUExact{});
}

void Activation::silu(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SiLU{});
}

void Activation::silu_fast(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SiLUFast{});
}

void Activation::swish(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Swish{beta});
}

void Activation::swish_fast(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4* Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SwishFast{beta});
}