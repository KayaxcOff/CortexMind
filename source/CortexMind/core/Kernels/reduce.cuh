//
// Created by muham on 30.03.2026.
//

#ifndef CORTEXMIND_CORE_KERNELS_REDUCE_CUH
#define CORTEXMIND_CORE_KERNELS_REDUCE_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utilities.cuh>
#include <CortexMind/core/Tools/utilities.hpp>

namespace cortex::_fw::cuda::kernels {

__device__ inline f32 warp_reduce_sum(f32 val) {
    for (i32 offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ inline f32 warp_reduce_max(f32 val) {
    for (i32 offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ inline f32 warp_reduce_min(f32 val) {
    for (i32 offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template<typename WarpFn>
__device__ inline f32 block_reduce(f32 val, WarpFn warp_fn, f32 identity) {
    __shared__ f32 shared[WARP_SIZE];
    const i32 lane   = threadIdx.x % WARP_SIZE;
    const i32 warp_id = threadIdx.x / WARP_SIZE;

    val = warp_fn(val);

    if (lane == 0)
        shared[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : identity;
    if (warp_id == 0)
        val = warp_fn(val);

    return val;
}

__global__ void reduce_sum_kernel(
    const bf16* __restrict Xx,
          f32*  __restrict out,
    size_t N)
{
    f32 val = 0.0f;
    CXM_CUDA_LOOP_1D(i, N)
        val += to_f32(Xx[i]);

    val = block_reduce(val,
        [](f32 v) { return warp_reduce_sum(v); }, 0.0f);

    if (threadIdx.x == 0)
        atomicAdd(out, val);
}

__global__ void reduce_max_kernel(
    const bf16* __restrict Xx,
          f32*  __restrict out,
    size_t N)
{
    f32 val = -CXM_F32_MAX;
    CXM_CUDA_LOOP_1D(i, N)
        val = fmaxf(val, to_f32(Xx[i]));

    val = block_reduce(val,
        [](f32 v) { return warp_reduce_max(v); }, -CXM_F32_MAX);

    if (threadIdx.x == 0)
        atomicMax(reinterpret_cast<int*>(out),
                  __float_as_int(val));
}

__global__ void reduce_min_kernel(
    const bf16* __restrict Xx,
          f32*  __restrict out,
    size_t N)
{
    f32 val = CXM_F32_MAX;
    CXM_CUDA_LOOP_1D(i, N)
        val = fminf(val, to_f32(Xx[i]));

    val = block_reduce(val,
        [](f32 v) { return warp_reduce_min(v); }, CXM_F32_MAX);

    if (threadIdx.x == 0)
        atomicMin(reinterpret_cast<int*>(out),
                  __float_as_int(val));
}

__global__ void reduce_variance_kernel(
    const bf16* __restrict Xx,
          f32   mean,
          f32*  __restrict out,
    size_t N)
{
    f32 val = 0.0f;
    CXM_CUDA_LOOP_1D(i, N) {
        const f32 diff = to_f32(Xx[i]) - mean;
        val += diff * diff;
    }

    val = block_reduce(val,
        [](f32 v) { return warp_reduce_sum(v); }, 0.0f);

    if (threadIdx.x == 0)
        atomicAdd(out, val);
}

} // namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_KERNELS_REDUCE_CUH