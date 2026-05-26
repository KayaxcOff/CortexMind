//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_REDUCE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_REDUCE_CUH

#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/loop.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <cmath>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief Performs warp-level sum reduction using shuffle instructions.
     *
     * @param val Input value per thread.
     * @return Sum of all values in the warp (valid only for lane 0).
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
     * @brief Performs warp-level maximum reduction using shuffle instructions.
     *
     * @param val Input value per thread.
     * @return Maximum value in the warp (valid only for lane 0).
     */
    __device__ inline f32 warp_reduce_max(f32 val) {
        val = fmaxf(val, shfl::down(val, 16));
        val = fmaxf(val, shfl::down(val, 8));
        val = fmaxf(val, shfl::down(val, 4));
        val = fmaxf(val, shfl::down(val, 2));
        val = fmaxf(val, shfl::down(val, 1));
        return val;
    }

    /**
     * @brief CUDA kernel for parallel sum reduction.
     *
     * Computes `sum(Xx[0..N-1])` and adds the result to `*Xz` using atomic operation.
     *
     * @param Xx Input array
     * @param Xz Output pointer (single float, accumulated via atomicAdd)
     * @param N  Number of elements
     *
     * @note Uses warp shuffle + shared memory reduction.
     * @note Multiple blocks can be launched; result is accumulated atomically.
     */
    __global__ void ReduceSum(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local = 0.0f;

        CXM_CUDA_LOOP_1D(i, N) {
            local += Xx[i];
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
                atomic::add<f32>(Xz, warp_val);
            }
        }
    }

    /**
     * @brief CUDA kernel for parallel variance reduction.
     *
     * Computes `sum((x[i] - mean)^2)` for i in [0, N) and adds the result
     * to `*Xz` using atomic operation.
     *
     * @param Xx  Input array
     * @param mean Pre-computed mean value
     * @param Xz  Output pointer (single float, accumulated via atomicAdd)
     * @param N   Number of elements
     *
     * @note Used as the second step in variance / standard deviation calculation.
     */
    __global__ void ReduceVar(const f32* __restrict Xx, const f32 mean, f32* __restrict Xz, const size_t N) {
        extern __shared__ f32 sdata[];

        const size_t tid = threadIdx.x;
        f32 local = 0.0f;

        CXM_CUDA_LOOP_1D(i, N) {
            const f32 diff = Xx[i] - mean;
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
                atomic::add<f32>(Xz, warp_val);
            }
        }
    }

    __global__ void ReduceSumFirstDim(const f32* __restrict__ Xx, f32* __restrict__ Xz, const size_t rows, const size_t cols) {
        extern __shared__ f32 sdata[];

        const size_t col = blockIdx.x;
        const size_t tid = threadIdx.x;

        if (col >= cols) return;

        f32 local = 0.0f;
        for (size_t r = tid; r < rows; r += blockDim.x) {
            local += Xx[r * cols + col];
        }

        // Warp reduce
        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }
        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 val = sdata[tid];
            val = warp_reduce_sum(val);
            if (tid == 0) {
                Xz[col] = val;
            }
        }
    }

    __global__ void ReduceSumLastDim(const f32* __restrict__ Xx, f32* __restrict__ Xz, const size_t rows, const size_t cols) {
        extern __shared__ f32 sdata[];

        const size_t row = blockIdx.x;
        const size_t tid = threadIdx.x;

        if (row >= rows) return;

        f32 local = 0.0f;
        for (size_t c = tid; c < cols; c += blockDim.x) {
            local += Xx[row * cols + c];
        }

        local = warp_reduce_sum(local);

        if (tid % WARP_SIZE == 0) {
            sdata[tid / WARP_SIZE] = local;
        }
        SynchronizeThreads();

        const size_t warp_count = blockDim.x / WARP_SIZE;
        if (tid < warp_count) {
            f32 val = sdata[tid];
            val = warp_reduce_sum(val);
            if (tid == 0) {
                Xz[row] = val;
            }
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_REDUCE_CUH