//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_SCALAR_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_SCALAR_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/loop.h>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief CUDA kernel for element-wise scalar operations using float4 vectorization.
     *
     * This templated kernel performs operations such as add, sub, mul, div between
     * a vector and a scalar value. It uses `float4` (f32x4) for vectorized processing
     * to achieve higher memory bandwidth and performance.
     *
     * @tparam OpType Functor type that defines the scalar operation (e.g. add, mul, etc.)
     *
     * @param Xx    Input array (vectorized as f32x4)
     * @param value Scalar value to apply
     * @param Xz    Output array (vectorized as f32x4)
     * @param N     Total number of elements (not necessarily multiple of 4)
     */
    template <typename OpType>
    __global__ void scalar(const f32x4* __restrict Xx, const f32 value, f32x4* __restrict Xz, const size_t N) {
        OpType op;

        const size_t vecN = N / 4;
        const size_t tail_start = vecN * 4;

        CXM_CUDA_LOOP_1D(i, vecN) {
            Xz[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value),
                op(Xx[i].z, value), op(Xx[i].w, value)
            };
        }

        const f32* Xx_f = reinterpret_cast<const f32*>(Xx);
        f32* Xz_f = reinterpret_cast<f32*>(Xz);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xz_f[i] = op(Xx_f[i], value);
        }
    }

    /**
     * @brief CUDA kernel for in-place element-wise scalar operations using float4 vectorization.
     *
     * Performs operations such as add, sub, mul, div between a vector and a scalar value,
     * then writes the result back to the input array (Xx = op(Xx, value)).
     *
     * @tparam OpType Functor type that defines the scalar operation
     *
     * @param Xx    Input/Output array (modified in-place, vectorized as f32x4)
     * @param value Scalar value to apply to each element
     * @param N     Total number of elements
     */
    template <typename OpType>
    __global__ void scalar_inplace(f32x4* __restrict Xx, const f32 value, const size_t N) {
        OpType op;

        const size_t vecN       = N / 4;
        const size_t tail_start = vecN * 4;

        CXM_CUDA_LOOP_1D(i, vecN) {
            Xx[i] = {
                op(Xx[i].x, value), op(Xx[i].y, value),
                op(Xx[i].z, value), op(Xx[i].w, value)
            };
        }

        f32* Xx_f = reinterpret_cast<f32*>(Xx);
        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xx_f[i] = op(Xx_f[i], value);
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_SCALAR_CUH