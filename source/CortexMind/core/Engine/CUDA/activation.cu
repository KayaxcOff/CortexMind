//
// Created by muham on 10.04.2026.
//

#include "CortexMind/core/Engine/CUDA/activation.h"
#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Engine/CUDA/Kernels/activation.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/operations.h>

using namespace cortex::_fw::cuda;

void Activation::relu(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::ReLU><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::ReLU{});
}

void Activation::leaky_relu(const f32* Xx, f32* Xz, const size_t N, const f32 alpha) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::LeakyReLU><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::LeakyReLU{alpha});
}

void Activation::sigmoid(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Sigmoid><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Sigmoid{});
}

void Activation::sigmoid_fast(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::SigmoidFast><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SigmoidFast{});
}

void Activation::tanh(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Tanh><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Tanh{});
}

void Activation::gelu(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::GELU><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::GELU{});
}

void Activation::gelu_exact(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::GELUExact><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::GELUExact{});
}

void Activation::silu(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::SiLU><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SiLU{});
}

void Activation::silu_fast(const f32* Xx, f32* Xz, const size_t N) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::SiLUFast><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SiLUFast{});
}

void Activation::swish(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::Swish><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::Swish{beta});
}

void Activation::swish_fast(const f32* Xx, f32* Xz, const size_t N, const f32 beta) {
    const f32x4* Xx4 = reinterpret_cast<const f32x4*>(Xx);
    f32x4*       Xz4 = reinterpret_cast<f32x4*>(Xz);
    kernels::activation<ops::SwishFast><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx4, Xz4, N, ops::SwishFast{beta});
}