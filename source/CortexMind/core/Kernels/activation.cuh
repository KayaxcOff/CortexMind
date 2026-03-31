//
// Created by muham on 31.03.2026.
//

#ifndef CORTEXMIND_CORE_KERNELS_ACTIVATION_CUH
#define CORTEXMIND_CORE_KERNELS_ACTIVATION_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utilities.cuh>

namespace cortex::_fw::cuda::kernels {
    template<typename Op>
    __global__ void activation(
        const bf16* __restrict Xx,
              bf16* __restrict Xz,
        size_t N, Op op)
    {
        CXM_CUDA_LOOP_1D(i, N) {
            Xz[i] = to_bf16(op(to_f32(Xx[i])));
        }
    }

    template<typename Op>
    __global__ void inplace_activation(bf16* Xx, size_t N, Op op) {
        CXM_CUDA_LOOP_1D(i, N) {
            Xx[i] = to_bf16(op(to_f32(Xx[i])));
        }
    }

    __global__ void softmax_scale(
              bf16* __restrict Xx,
        const f32*  __restrict max_val,
        const f32*  __restrict sum_val,
        size_t N)
    {
        CXM_CUDA_LOOP_1D(i, N) {
            const f32 e = expf(to_f32(Xx[i]) - *max_val);
            Xx[i] = to_bf16(e / *sum_val);
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_KERNELS_ACTIVATION_CUH