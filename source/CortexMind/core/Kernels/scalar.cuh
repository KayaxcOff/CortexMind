//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_KERNELS_SCALAR_CUH
#define CORTEXMIND_CORE_KERNELS_SCALAR_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utilities.cuh>

namespace cortex::_fw::cuda::kernels {
    template<typename Op>
    __global__ void scalar(const bf16* __restrict Xx, const f32 value, bf16* __restrict Xz, const size_t N) {
        CXM_CUDA_LOOP_1D(i, N) {
            const f32 x = to_f32(Xx[i]);
            const f32 z = Op{}(x, value);
            Xz[i] = to_bf16(z);
        }
    }

    template<typename Op>
    __global__ void inplace_scalar(bf16* Xx, const f32 value, const size_t N) {
        CXM_CUDA_LOOP_1D(i, N) {
            const f32 x = to_f32(Xx[i]);
            const f32 z = Op{}(x, value);
            Xx[i] = to_bf16(z);
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_KERNELS_SCALAR_CUH