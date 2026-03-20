//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Engine/CUDA/reduce.cuh"
#include <CortexMind/core/Engine/CUDA/cast.cuh>
#include <CortexMind/core/Engine/CUDA/reduce_kernel.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>
#include <CortexMind/core/Engine/Memory/transform.cuh>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

reduce_t::reduce_t() {
    f32* raw = nullptr;
    cudaMalloc(&raw, sizeof(f32));
    d_tmp.reset(raw);
}

f32 reduce_t::hsum(const f32* Xx, const size_t idx) {
    if (idx == 0) return 0.0f;
    cudaMemset(d_tmp.get(), 0, sizeof(f32));
    const size_t vec_count = (idx + 3) / 4;
    kernels::reduce_hsum_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), d_tmp.get(), idx);
    CXM_CUDA_CHECK();
    f32 result = 0.0f;
    sys::transform<f32>::download(&result, d_tmp.get(), 1);
    return result;
}

f32 reduce_t::hmax(const f32* Xx, const size_t idx) {
    if (idx == 0) return 0.0f;
    //const size_t vec_count = (idx + 3) / 4;
    kernels::reduce_hmax_kernel<<<1, BLOCK_SIZE_1D>>>(to_vec(Xx), d_tmp.get(), idx);
    CXM_CUDA_CHECK();
    f32 result = 0.0f;
    sys::transform<f32>::download(&result, d_tmp.get(), 1);
    return result;
}

f32 reduce_t::hmin(const f32* Xx, const size_t idx) {
    if (idx == 0) return 0.0f;
    //const size_t vec_count = (idx + 3) / 4;
    kernels::reduce_hmin_kernel<<<1, BLOCK_SIZE_1D>>>(to_vec(Xx), d_tmp.get(), idx);
    CXM_CUDA_CHECK();
    f32 result = 0.0f;
    sys::transform<f32>::download(&result, d_tmp.get(), 1);
    return result;
}

f32 reduce_t::mean(const f32* Xx, const size_t idx) {
    if (idx == 0) return 0.0f;
    cudaMemset(d_tmp.get(), 0, sizeof(f32));
    const size_t vec_count = (idx + 3) / 4;
    kernels::reduce_mean_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(to_vec(Xx), d_tmp.get(), idx);
    CXM_CUDA_CHECK();
    f32 result = 0.0f;
    sys::transform<f32>::download(&result, d_tmp.get(), 1);
    return result / static_cast<f32>(idx);
}

void reduce_t::softmax(const f32* __restrict__ Xx, f32* __restrict__ Xz, const size_t idx) {
    if (idx == 0) return;
    kernels::reduce_softmax_kernel<<<1, BLOCK_SIZE_1D>>>(to_vec(Xx), to_vec(Xz), idx);
    CXM_CUDA_CHECK();
}

f32 reduce_t::sum(const f32* Xx, const size_t idx) {
    if (idx == 0) return 0.0f;
    cudaMemset(d_tmp.get(), 0, sizeof(f32));
    const size_t vec_count = (idx + 3) / 4;
    kernels::reduce_hsum_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(
        to_vec(Xx), d_tmp.get(), idx);
    CXM_CUDA_CHECK();
    f32 result = 0.0f;
    sys::transform<f32>::download(&result, d_tmp.get(), 1);
    return result;
}

f32 reduce_t::var(const f32* Xx, const size_t idx) {
    if (idx == 0) return 0.0f;
    const f32 m = mean(Xx, idx);
    cudaMemset(d_tmp.get(), 0, sizeof(f32));
    const size_t vec_count = (idx + 3) / 4;
    kernels::reduce_var_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(
        to_vec(Xx), m, d_tmp.get(), idx);
    CXM_CUDA_CHECK();
    f32 result = 0.0f;
    sys::transform<f32>::download(&result, d_tmp.get(), 1);
    return result / static_cast<f32>(idx);
}

f32 reduce_t::norm(const f32* Xx, const size_t idx) {
    if (idx == 0) return 0.0f;
    cudaMemset(d_tmp.get(), 0, sizeof(f32));
    const size_t vec_count = (idx + 3) / 4;
    kernels::reduce_norm_kernel<<<grid1d(vec_count), BLOCK_SIZE_1D>>>(
        to_vec(Xx), d_tmp.get(), idx);
    CXM_CUDA_CHECK();
    f32 result = 0.0f;
    sys::transform<f32>::download(&result, d_tmp.get(), 1);
    return std::sqrt(result);
}

void reduce_t::sum_dim(const f32* __restrict__ Xx, f32* __restrict__ Xz,
                       const size_t outer, const size_t inner, const size_t after) {
    if (outer == 0 || inner == 0 || after == 0) return;
    const size_t blocks = outer * after;
    kernels::reduce_sum_dim_kernel<<<blocks, BLOCK_SIZE_1D>>>(Xx, Xz, outer, inner, after);
    CXM_CUDA_CHECK();
}