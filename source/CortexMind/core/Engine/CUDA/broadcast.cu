//
// Created by muham on 17.03.2026.
//

#include "CortexMind/core/Engine/CUDA/broadcast.cuh"
#include <CortexMind/core/Engine/CUDA/broadcast_kernel.cuh>
#include <CortexMind/core/Engine/CUDA/op.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

// shape/stride vektörlerini GPU'ya kopyalar, işlem biter bitmez serbest bırakır
static i64* upload_vec(const std::vector<i64>& v) {
    i64* d_ptr = nullptr;
    CXM_ASSERT(cudaMalloc(&d_ptr, v.size() * sizeof(i64)) == cudaSuccess,
        "broadcast_t::upload_vec()", "cudaMalloc failed");
    CXM_ASSERT(cudaMemcpy(d_ptr, v.data(), v.size() * sizeof(i64), cudaMemcpyHostToDevice) == cudaSuccess,
        "broadcast_t::upload_vec()", "cudaMemcpy failed");
    return d_ptr;
}

template<typename Op>
static void launch_impl(
    const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
    const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
    const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
    const std::vector<i64>& shape_out)
{
    const i64 ndim   = static_cast<i64>(shape_out.size());
    size_t    numel  = 1;
    for (const i64 d : shape_out) numel *= static_cast<size_t>(d);

    // shape ve stride'ları GPU'ya yükle
    i64* d_shape_x   = upload_vec(shape_x);
    i64* d_stride_x  = upload_vec(stride_x);
    i64* d_shape_y   = upload_vec(shape_y);
    i64* d_stride_y  = upload_vec(stride_y);
    i64* d_shape_out = upload_vec(shape_out);

    kernels::broadcast_kernel<Op><<<grid1d(numel), BLOCK_SIZE_1D>>>(
        Xx, Xy, Xz,
        d_shape_x, d_stride_x,
        d_shape_y, d_stride_y,
        d_shape_out, ndim, numel);
    CXM_CUDA_CHECK();

    cudaFree(d_shape_x);
    cudaFree(d_stride_x);
    cudaFree(d_shape_y);
    cudaFree(d_stride_y);
    cudaFree(d_shape_out);
}

void broadcast_t::add(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                      const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                      const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                      const std::vector<i64>& shape_out) {
    launch_impl<op::Add>(Xx, Xy, Xz, shape_x, stride_x, shape_y, stride_y, shape_out);
}

void broadcast_t::sub(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                      const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                      const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                      const std::vector<i64>& shape_out) {
    launch_impl<op::Sub>(Xx, Xy, Xz, shape_x, stride_x, shape_y, stride_y, shape_out);
}

void broadcast_t::mul(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                      const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                      const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                      const std::vector<i64>& shape_out) {
    launch_impl<op::Mul>(Xx, Xy, Xz, shape_x, stride_x, shape_y, stride_y, shape_out);
}

void broadcast_t::div(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                      const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                      const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                      const std::vector<i64>& shape_out) {
    launch_impl<op::Div>(Xx, Xy, Xz, shape_x, stride_x, shape_y, stride_y, shape_out);
}