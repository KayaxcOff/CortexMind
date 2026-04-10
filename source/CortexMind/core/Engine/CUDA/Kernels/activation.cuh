//
// Created by muham on 10.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_ACTIVATION_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_ACTIVATION_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/loop.h>
#include <CortexMind/core/Tools/operations.h>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief CUDA kernel for element-wise activation functions using float4 vectorization.
     *
     * This templated kernel applies a unary activation function (such as ReLU, Sigmoid,
     * GELU, Tanh, etc.) to each element of the input array. It uses `float4` (f32x4)
     * vectorization for better memory bandwidth and performance.
     *
     * @tparam OpType Functor type that defines the activation operation (e.g. ops::ReLU, ops::Sigmoid, etc.)
     *
     * @param Xx  Input array (vectorized as f32x4)
     * @param Xz  Output array (vectorized as f32x4)
     * @param N   Total number of elements (not necessarily multiple of 4)
     * @param op  Activation functor (passed by value)
     */
    template <typename OpType>
    __global__ void activation(const f32x4* __restrict Xx, f32x4* __restrict Xz, const size_t N, const OpType op) {
        const size_t vecN       = N / 4;
        const size_t tail_start = vecN * 4;

        CXM_CUDA_LOOP_1D(i, vecN) {
            Xz[i] = {
                op(Xx[i].x), op(Xx[i].y),
                op(Xx[i].z), op(Xx[i].w)
            };
        }

        const f32* Xx_f = reinterpret_cast<const f32*>(Xx);
        f32* Xz_f = reinterpret_cast<f32*>(Xz);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xz_f[i] = op(Xx_f[i]);
        }
    }

} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_ACTIVATION_CUH