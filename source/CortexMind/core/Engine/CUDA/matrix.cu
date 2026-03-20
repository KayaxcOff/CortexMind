//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Engine/CUDA/matrix.cuh"
#include <CortexMind/core/Engine/CUDA/cast.cuh>
#include <CortexMind/core/Engine/CUDA/matrix_kernel.cuh>
#include <CortexMind/core/Engine/CUDA/op.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

void matrix_t::add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::matrix_kernel<op::Add><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), to_vec(Xz), idx);
    CXM_CUDA_CHECK();
}

void matrix_t::sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::matrix_kernel<op::Sub><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), to_vec(Xz), idx);
}

void matrix_t::mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::matrix_kernel<op::Mul><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), to_vec(Xz), idx);
}

void matrix_t::div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::matrix_kernel<op::Div><<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), to_vec(Xz), idx);
}

void matrix_t::fma(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, f32* __restrict Xk, size_t idx) {
     if (idx == 0) return;
     const size_t vec_count = (idx + 3) / 4;
     kernels::matrix_fma_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xy), to_vec(Xz), to_vec(Xk), idx);
     CXM_CUDA_CHECK();
}

void matrix_t::matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const size_t xIdx, const size_t yIdx, const size_t zIdx) {
    cublasHandle_t handle = get_handle();

    f32 alpha = 1.0f;
    f32 beta  = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)zIdx, (int)xIdx, (int)yIdx,
        &alpha,
        Xy, CUDA_R_32F, (int)zIdx,
        Xx, CUDA_R_32F, (int)yIdx,
        &beta,
        Xz, CUDA_R_32F, (int)zIdx,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    CXM_ASSERT(status == CUBLAS_STATUS_SUCCESS, "matrix_t::matmul", "cuBLAS GemmEx failed!");
}

cublasHandle_t matrix_t::get_handle() {
    struct HandleGuard {
        cublasHandle_t h;
        HandleGuard() {
            CXM_ASSERT(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "matrix_t::get_handle()", "Failed to create cuBLAS handle");
            cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        }
        ~HandleGuard() { cublasDestroy(h); }
    };
    static HandleGuard guard;
    return guard.h;
}

void matrix_t::fill(f32* __restrict__ Xx, const f32 value, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::matrix_fill_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), value, idx);
    CXM_CUDA_CHECK();
}

void matrix_t::sqrt(const f32* __restrict__ Xx, f32* __restrict__ Xz, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::matrix_sqrt_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xz), idx);
    CXM_CUDA_CHECK();
}

void matrix_t::pow(const f32* __restrict__ Xx, const f32 value, f32* __restrict__ Xz, const size_t idx) {
    if (idx == 0) return;
    const size_t vec_count = (idx + 3) / 4;
    kernels::matrix_pow_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), value, to_vec(Xz), idx);
    CXM_CUDA_CHECK();
}