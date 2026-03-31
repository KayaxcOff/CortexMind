//
// Created by muham on 29.03.2026.
//

#include "CortexMind/core/Engine/CUDA/matrix.h"
#include <CortexMind/core/Kernels/matrix.cuh>
#include <CortexMind/core/Tools/ops.h>
#include <CortexMind/core/Tools/utilities.hpp>
#include <cublasLt.h>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

namespace {
    template<typename Op>
    void dispatch_binary(
        const bf16* Xx, const bf16* Xy, bf16* Xz, size_t N)
    {
        if (N == 0) return;
        const size_t vec_N = N / 2;
        if (vec_N > 0)
            kernels::matrix<Op><<<grid1d(vec_N), BLOCK_SIZE_1D>>>(Xx, Xy, Xz, N);
        if (N % 2 != 0)
            kernels::matrix_tail<Op><<<1, 1>>>(Xx, Xy, Xz, N - 1);
    }

    template<typename Op>
    void dispatch_inplace(bf16* Xx, const bf16* Xy, size_t N) {
        if (N == 0) return;
        const size_t vec_N = N / 2;
        if (vec_N > 0)
            kernels::inplace_matrix<Op><<<grid1d(vec_N), BLOCK_SIZE_1D>>>(Xx, Xy, N);
        if (N % 2 != 0)
            kernels::inplace_matrix_tail<Op><<<1, 1>>>(Xx, Xy, N - 1);
    }
}

void Matrix::add(const bf16* Xx, const bf16* Xy, bf16* Xz, size_t N) { dispatch_binary<ops::Add>(Xx, Xy, Xz, N); }
void Matrix::sub(const bf16* Xx, const bf16* Xy, bf16* Xz, size_t N) { dispatch_binary<ops::Sub>(Xx, Xy, Xz, N); }
void Matrix::mul(const bf16* Xx, const bf16* Xy, bf16* Xz, size_t N) { dispatch_binary<ops::Mul>(Xx, Xy, Xz, N); }
void Matrix::div(const bf16* Xx, const bf16* Xy, bf16* Xz, size_t N) { dispatch_binary<ops::Div>(Xx, Xy, Xz, N); }

void Matrix::add(bf16* Xx, const bf16* Xy, size_t N) { dispatch_inplace<ops::Add>(Xx, Xy, N); }
void Matrix::sub(bf16* Xx, const bf16* Xy, size_t N) { dispatch_inplace<ops::Sub>(Xx, Xy, N); }
void Matrix::mul(bf16* Xx, const bf16* Xy, size_t N) { dispatch_inplace<ops::Mul>(Xx, Xy, N); }
void Matrix::div(bf16* Xx, const bf16* Xy, size_t N) { dispatch_inplace<ops::Div>(Xx, Xy, N); }

