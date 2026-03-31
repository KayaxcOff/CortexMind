//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_KERNELS_MATRIX_CUH
#define CORTEXMIND_CORE_TOOLS_KERNELS_MATRIX_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/core/Tools/utilities.cuh>

namespace cortex::_fw::cuda::kernels {

    template<typename Op>
    __global__ void matrix(
        const bf16* __restrict Xx,
        const bf16* __restrict Xy,
              bf16* __restrict Xz,
        const size_t N)
    {
        const size_t vec_N = N / 2;
        const auto* Xx2 = reinterpret_cast<const bf2x16*>(Xx);
        const auto* Xy2 = reinterpret_cast<const bf2x16*>(Xy);
              auto* Xz2 = reinterpret_cast<bf2x16*>(Xz);


        CXM_CUDA_LOOP_1D(i, vec_N) {
            const bf2x16 vx = Xx2[i];
            const bf2x16 vy = Xy2[i];
            Xz2[i] = to_bf162(
                Op{}(bf162_lo(vx), bf162_lo(vy)),
                Op{}(bf162_hi(vx), bf162_hi(vy))
            );
        }
    }

    template<typename Op>
    __global__ void matrix_tail(
        const bf16* __restrict Xx,
        const bf16* __restrict Xy,
              bf16* __restrict Xz,
        const size_t idx)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            Xz[idx] = to_bf16(Op{}(to_f32(Xx[idx]), to_f32(Xy[idx])));
        }
    }

    template<typename Op>
    __global__ void inplace_matrix(
              bf16* __restrict Xx,
        const bf16* __restrict Xy,
        const size_t N)
    {
        const size_t vec_N = N / 2;
        auto*       Xx2 = reinterpret_cast<bf2x16*>(Xx);
        const auto* Xy2 = reinterpret_cast<const bf2x16*>(Xy);

        CXM_CUDA_LOOP_1D(i, vec_N) {
            const bf2x16 vx = Xx2[i];
            const bf2x16 vy = Xy2[i];
            Xx2[i] = to_bf162(
                Op{}(bf162_lo(vx), bf162_lo(vy)),
                Op{}(bf162_hi(vx), bf162_hi(vy))
            );
        }
    }

    template<typename Op>
    __global__ void inplace_matrix_tail(bf16* Xx, const bf16* __restrict Xy, const size_t idx) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            Xx[idx] = to_bf16(Op{}(to_f32(Xx[idx]), to_f32(Xy[idx])));
        }
    }

} // namespace cortex::_fw::cuda::kernels

#endif // CORTEXMIND_CORE_TOOLS_KERNELS_MATRIX_CUH