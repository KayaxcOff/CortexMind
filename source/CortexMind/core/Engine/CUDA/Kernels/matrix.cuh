//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_MATRIX_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_MATRIX_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/broadcast.hpp>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/loop.h>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief CUDA kernel for element-wise binary matrix/vector operations using float4 vectorization.
     *
     * Performs operations such as add, sub, mul, div between two arrays (Xx and Xy)
     * and writes the result to a third array (Xz). Uses `float4` (f32x4) for better
     * memory bandwidth and performance.
     *
     * @tparam OpType Functor type that defines the binary operation (e.g. ops::Addition, ops::Multiplication, etc.)
     *
     * @param Xx  First input array (vectorized as f32x4)
     * @param Xy  Second input array (vectorized as f32x4)
     * @param Xz  Output array (vectorized as f32x4)
     * @param N   Total number of elements (not necessarily multiple of 4)
     */
    template <typename OpType>
    __global__ void matrix(const f32x4* __restrict Xx, const f32x4* __restrict Xy, f32x4* __restrict Xz, const size_t N) {
        OpType op;

        const size_t vecN = N / 4;
        const size_t tail_start = vecN * 4;

        CXM_CUDA_LOOP_1D(i, vecN) {
            Xz[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
                op(Xx[i].z, Xy[i].z), op(Xx[i].w, Xy[i].w)
            };
        }

        const f32* Xx_f = reinterpret_cast<const f32*>(Xx);
        const f32* Xy_f = reinterpret_cast<const f32*>(Xy);
        f32* Xz_f = reinterpret_cast<f32*>(Xz);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xz_f[i] = op(Xx_f[i], Xy_f[i]);
        }
    }

    /**
     * @brief CUDA kernel for in-place element-wise binary operations using float4 vectorization.
     *
     * Performs operations such as add, sub, mul, div between two arrays and writes
     * the result back to the first array (Xx = op(Xx, Xy)).
     *
     * @tparam OpType Functor type that defines the binary operation
     *
     * @param Xx  Input/Output array (modified in-place, vectorized as f32x4)
     * @param Xy  Second input array (vectorized as f32x4)
     * @param N   Total number of elements
     */
    template <typename OpType>
    __global__ void matrix_inplace(f32x4* __restrict Xx, const f32x4* __restrict Xy, const size_t N) {
        OpType op;

        const size_t vecN       = N / 4;
        const size_t tail_start = vecN * 4;

        CXM_CUDA_LOOP_1D(i, vecN) {
            Xx[i] = {
                op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
                op(Xx[i].z, Xy[i].z), op(Xx[i].w, Xy[i].w)
            };
        }

        f32* Xx_f = reinterpret_cast<f32*>(Xx);
        const f32* Xy_f = reinterpret_cast<const f32*>(Xy);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xx_f[i] = op(Xx_f[i], Xy_f[i]);
        }
    }

    template <typename OpType>
    __global__ void matrix_broadcast(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const size_t N, const BroadcastInfo info) {
        OpType op;

        CXM_CUDA_LOOP_1D(i, N) {
            size_t offset_x   = 0;
            size_t offset_y   = 0;
            size_t offset_z   = 0;
            size_t linear_idx = i;

            for (int d = info.ndim - 1; d >= 0; --d) {
                const size_t coord = linear_idx % info.shape[d];
                offset_x  += coord * info.stride_x[d];
                offset_y  += coord * info.stride_y[d];
                offset_z  += coord * info.stride_z[d];
                linear_idx /= info.shape[d];
            }

            Xz[offset_z] = op(Xx[offset_x], Xy[offset_y]);
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_MATRIX_CUH