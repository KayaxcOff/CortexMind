//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_MATRIX_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_MATRIX_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utils.cuh>
#include <CortexMind/core/Tools/loop.h>

namespace cortex::_fw::cuda::kernels {
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
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_KERNELS_MATRIX_CUH