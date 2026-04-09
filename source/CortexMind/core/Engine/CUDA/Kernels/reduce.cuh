//
// Created by muham on 9.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_REDUCE_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_REDUCE_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/loop.h>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief Performs warp-level sum reduction using shuffle instructions.
     *
     * This function reduces 32 threads (one warp) to a single value using
     * `__shfl_down_sync` operations. It is used as a building block for block-level reductions.
     *
     * @param val Per-thread input value
     * @return Sum of the warp (valid only in lane 0)
     */
    __device__ inline f32 warp_reduce_sum(f32 val) {
        val += shfl::down(val, 16);
        val += shfl::down(val, 8);
        val += shfl::down(val, 4);
        val += shfl::down(val, 2);
        val += shfl::down(val, 1);
        return val;
    }

    /**
     * @brief CUDA kernel for parallel sum reduction.
     *
     * Computes the sum of all elements in the input array using a highly optimized
     * warp + shared memory reduction strategy with atomic add to global memory.
     *
     * @param x     Input array
     * @param out   Output pointer (single float, accumulated atomically)
     * @param N     Number of elements in the array
     */
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

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_sum(warp_val);
            if (tid == 0) {
                atomic::add<f32>(out, warp_val);
            }
        }
    }

    /**
     * @brief CUDA kernel for parallel variance reduction.
     *
     * Computes the sum of squared differences from the mean `(x[i] - mean)²`
     * using the same warp + shared memory reduction technique.
     *
     * @param x     Input array
     * @param mean  Pre-computed mean value
     * @param out   Output pointer (single float, accumulated atomically)
     * @param N     Number of elements in the array
     */
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

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_sum(warp_val);
            if (tid == 0) {
                atomic::add<f32>(out, warp_val);
            }
        }
    }

} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_REDUCE_CUH