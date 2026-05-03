//
// Created by muham on 29.04.2026.
//

#include "CortexMind/core/Engine/CUDA/matrix_broadcast.h"
#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Engine/CUDA/Kernels/matrix.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/operations.h>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/runtime/provider.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void MatrixBroadcast::add(const f32* Xx, const f32* Xy, f32* Xz, const size_t N, const BroadcastInfo& info_ptr) {
    kernels::matrix_broadcast<ops::Addition><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xy, Xz, N, info_ptr);
}

void MatrixBroadcast::sub(const f32* Xx, const f32* Xy, f32* Xz, const size_t N, const BroadcastInfo& info_ptr) {
    kernels::matrix_broadcast<ops::Subtraction><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xy, Xz, N, info_ptr);
}

void MatrixBroadcast::mul(const f32* Xx, const f32* Xy, f32* Xz, const size_t N, const BroadcastInfo& info_ptr) {
    kernels::matrix_broadcast<ops::Multiplication><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xy, Xz, N, info_ptr);
}

void MatrixBroadcast::div(const f32* Xx, const f32* Xy, f32* Xz, const size_t N, const BroadcastInfo& info_ptr) {
    kernels::matrix_broadcast<ops::Division><<<grid1d(N), BLOCK_SIZE_1D>>>(Xx, Xy, Xz, N, info_ptr);
}