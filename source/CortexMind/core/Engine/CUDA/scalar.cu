//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/CUDA/scalar.h"
#include <CortexMind/core/Kernels/scalar.cuh>
#include <CortexMind/core/Tools/ops.h>
#include <CortexMind/core/Tools/utilities.hpp>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void ScalarKernel::add(const bf16* Xx, const f32 value, bf16* Xz, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::scalar<ops::Add><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, Xz, N);
}

void ScalarKernel::sub(const bf16* Xx, const f32 value, bf16* Xz, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::scalar<ops::Sub><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, Xz, N);
}

void ScalarKernel::mul(const bf16* Xx, const f32 value, bf16* Xz, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::scalar<ops::Mul><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, Xz, N);
}

void ScalarKernel::div(const bf16* Xx, const f32 value, bf16* Xz, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::scalar<ops::Div><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, Xz, N);
}

void ScalarKernel::add(bf16* Xx, const f32 value, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::inplace_scalar<ops::Add><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, N);
}

void ScalarKernel::sub(bf16* Xx, const f32 value, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::inplace_scalar<ops::Sub><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, N);
}

void ScalarKernel::mul(bf16* Xx, const f32 value, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::inplace_scalar<ops::Mul><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, N);
}

void ScalarKernel::div(bf16* Xx, const f32 value, const size_t N) {
    if (N == 0) {
        return;
    }
    kernels::inplace_scalar<ops::Div><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, value, N);
}