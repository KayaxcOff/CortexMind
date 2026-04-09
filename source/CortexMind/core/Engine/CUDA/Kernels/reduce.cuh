//
// Created by muham on 9.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_REDUCE_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_REDUCE_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/loop.h>

namespace cortex::_fw::cuda::kernels {

    __device__ inline f32 warp_reduce_sum(f32 val) {
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        return val;
    }

    __global__ void reduce_sum(const f32* __restrict x, f32* __restrict out, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local = 0.0f;

        CXM_CUDA_LOOP_1D(i, N) {
            local += x[i];
        }

        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0)
            sdata[tid / WARP_SIZE] = local;

        __syncthreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_sum(warp_val);
            if (tid == 0)
                atomicAdd(out, warp_val);
        }
    }

    __global__ void reduce_var(const f32* __restrict x, const f32 mean, f32* __restrict out, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local = 0.0f;

        CXM_CUDA_LOOP_1D(i, N) {
            const f32 diff = x[i] - mean;
            local += diff * diff;
        }

        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }

        __syncthreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_sum(warp_val);
            if (tid == 0) {
                atomicAdd(out, warp_val);
            }
        }
    }

} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_REDUCE_CUH