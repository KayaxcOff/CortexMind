//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_ACTIVATION_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_ACTIVATION_CUH

#include <CortexMind/framework/Engine/CUDA/types.cuh>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/loop.hpp>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief Generic element-wise activation kernel.
     *
     * Applies a unary activation function (`OpType`) to each element of the input array.
     *
     * @tparam OpType Functor type that implements `__device__ f32 operator()(f32)`
     *                (e.g. `ops::ReLU`, `ops::GELU`, `ops::SiLU`, etc.)
     *
     * @param Xx Input array (f32x4 aligned)
     * @param Xz Output array
     * @param N  Number of elements
     *
     * @note Uses 4-wide vectorized processing (`f32x4`) for the main loop
     *       and scalar fallback for the tail (remainder).
     */
    template<typename OpType>
    __global__ void activation(const f32x4* __restrict__ Xx, f32x4* __restrict__ Xz, const size_t N, const OpType op) {

        const size_t vecN = N >> 2;
        const size_t tail_start = vecN << 2;

        CXM_CUDA_LOOP_1D(i, vecN) {
            const f32x4 x = Xx[i];
            Xz[i] = {
                op(x.x), op(x.y),
                op(x.z), op(x.w)
            };
        }

        const f32* Xx_f = reinterpret_cast<const f32*>(Xx);
        f32* Xz_f = reinterpret_cast<f32*>(Xz);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xz_f[i] = op(Xx_f[i]);
        }
    }
} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_ACTIVATION_CUH