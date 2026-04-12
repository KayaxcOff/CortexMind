//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_SOFTMAX_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_SOFTMAX_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Engine/CUDA/Kernels/reduce.cuh>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/loop.h>
#include <cfloat>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief CUDA kernel that finds the maximum value in the input array (for softmax).
     *
     * Uses warp-level reduction with shared memory to compute the global maximum
     * efficiently. The result is written to the `out` pointer (single value).
     *
     * @param x     Input array
     * @param out   Output pointer to store the maximum value
     * @param N     Number of elements
     */
    __global__ void softmax_max(const f32* __restrict x, f32* __restrict out, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local = -FLT_MAX;

        CXM_CUDA_LOOP_1D(i, N) {
            local = fmaxf(local, x[i]);
        }

        local = warp_reduce_max(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_max(warp_val);
            if (tid == 0) {
                *out = warp_val;
            }
        }
    }

    /**
     * @brief CUDA kernel that normalizes the exponentials by dividing by their sum.
     *
     * Computes `x[i] = x[i] / sum` for all elements using fast reciprocal division.
     *
     * @param x     Input/Output array (contains exponentials, will be normalized in-place)
     * @param sum   Sum of exponentials (pre-computed)
     * @param N     Number of elements
     */
    __global__ void softmax_exp_sum(f32* __restrict x, const f32 x_max,
                                    f32* __restrict sum_out, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local_sum = 0.0f;

        CXM_CUDA_LOOP_1D(i, N) {
            const f32 e = __expf(x[i] - x_max);
            x[i] = e;
            local_sum += e;
        }

        local_sum = warp_reduce_sum(local_sum);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local_sum;
        }

        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 warp_val = sdata[tid];
            warp_val = warp_reduce_sum(warp_val);
            if (tid == 0) {
                atomic::add(sum_out, warp_val);
            }
        }
    }

    __global__ void softmax_normalize(f32* __restrict x, const f32 sum, const size_t N) {
        const f32 inv_sum = __fdividef(1.0f, sum);

        CXM_CUDA_LOOP_1D(i, N) {
            x[i] *= inv_sum;
        }
    }

} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_SOFTMAX_CUH