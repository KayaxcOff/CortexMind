//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_BROADCAST_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_BROADCAST_CUH

#include <CortexMind/framework/Engine/CUDA/types.cuh>
#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/loop.hpp>

namespace cortex::_fw::cuda::kernels {

    /**
     * @brief Row broadcast kernel — X(M,N) op Y(N) → Z(M,N)
     *
     * Y is applied to every row of X.
     * Each thread processes one element: (row, col).
     *
     * @tparam OpType Binary functor: __device__ f32 operator()(f32, f32)
     */
    template<typename OpType>
    __global__ void row_broadcast(
        const f32* __restrict__ Xx,
        const f32* __restrict__ Xy,
        f32* __restrict__       Xz,
        const size_t            M,
        const size_t            N,
        const OpType            op)
    {
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= M || col >= N) return;

        Xz[row * N + col] = op(Xx[row * N + col], Xy[col]);
    }

    /**
     * @brief Row broadcast in-place kernel — X(M,N) op= Y(N)
     */
    template<typename OpType>
    __global__ void row_broadcast_ip(
        f32* __restrict__       Xx,
        const f32* __restrict__ Xy,
        const size_t            M,
        const size_t            N,
        const OpType            op)
    {
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= M || col >= N) return;

        Xx[row * N + col] = op(Xx[row * N + col], Xy[col]);
    }

    /**
     * @brief Col broadcast kernel — X(M,N) op Y(M) → Z(M,N)
     *
     * Y[row] is broadcast across every element in row of X.
     * blockIdx.y == row, so gridDim.y must equal M exactly.
     * Y[row] is loaded once into shared memory per block.
     *
     * @note M must be < 65535 (CUDA gridDim.y limit).
     * @tparam OpType Binary functor: __device__ f32 operator()(f32, f32)
     */
    template<typename OpType>
    __global__ void col_broadcast(
        const f32* __restrict__ Xx,
        const f32* __restrict__ Xy,
        f32* __restrict__       Xz,
        const size_t            M,
        const size_t            N,
        const OpType            op)
    {
        __shared__ f32 y_shared;

        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t row = blockIdx.y;

        if (threadIdx.x == 0) {
            y_shared = Xy[row];
        }
        SynchronizeThreads();

        if (row >= M || col >= N) {
            return;
        }

        Xz[row * N + col] = op(Xx[row * N + col], y_shared);
    }

    /**
     * @brief Col broadcast in-place kernel — X(M,N) op= Y(M)
     *
     * @note M must be < 65535.
     */
    template<typename OpType>
    __global__ void col_broadcast_ip(f32* __restrict__ Xx,const f32* __restrict__ Xy, const size_t M, const size_t N, const OpType op) {
        __shared__ f32 y_shared;

        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t row = blockIdx.y;

        if (threadIdx.x == 0) {
            y_shared = Xy[row];
        }
        SynchronizeThreads();

        if (row >= M || col >= N) {
            return;
        }

        Xx[row * N + col] = op(Xx[row * N + col], y_shared);
    }

    template<typename OpType>
    __global__ void general_broadcast(
        const f32* __restrict__ Xx,
        const f32* __restrict__ Xy,
        f32* __restrict__       Xz,
        const BroadcastInfo     info,
        const size_t            total,
        const OpType            op)
    {
        CXM_CUDA_LOOP_1D(i, total) {
            size_t ox = 0, oy = 0, oz = 0;
            size_t idx = i;

            for (i32 d = info.ndim - 1; d >= 0; --d) {
                const size_t coord = idx % info.shape[d];
                ox  += coord * info.stride_x[d];
                oy  += coord * info.stride_y[d];
                oz  += coord * info.stride_z[d];
                idx /= info.shape[d];
            }

            Xz[oz] = op(Xx[ox], Xy[oy]);
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_BROADCAST_CUH