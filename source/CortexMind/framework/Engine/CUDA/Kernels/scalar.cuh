//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_SCALAR_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_SCALAR_CUH

#include <CortexMind/framework/Engine/CUDA/types.cuh>
#include <CortexMind/framework/Tools/loop.hpp>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief Generic CUDA kernel for element-wise scalar operation.
     *
     * Applies a scalar value to each element of the input array using the
     * provided operation functor (`OpType`).
     *
     * @tparam OpType Functor type (e.g. `ops::Addition`, `ops::Multiplication`, etc.)
     *
     * @param Xx    Input array (f32x4 aligned)
     * @param value Scalar value to apply
     * @param Xz    Output array
     * @param N     Number of elements
     *
     * @note Uses 4-wide vectorized processing (`f32x4`) + tail handling for
     *       maximum performance and correctness.
     */
    template<typename OpType>
    __global__ void scalar(const f32x4* __restrict__ Xx, const f32 value, f32x4* __restrict__ Xz, const size_t N) {
        const OpType op{};

        const size_t vecN = N >> 2;          // N / 4
        const size_t tail_start = vecN << 2;       // vecN * 4

        CXM_CUDA_LOOP_1D(i, vecN) {
            const f32x4 x = Xx[i];
            Xz[i] = {
                op(x.x, value), op(x.y, value),
                op(x.z, value), op(x.w, value)
            };
        }

        // Tail: remaining elements where N % 4 != 0
        const f32* __restrict__ Xx_f = reinterpret_cast<const f32*>(Xx);
        f32* __restrict__ Xz_f = reinterpret_cast<f32*>(Xz);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xz_f[i] = op(Xx_f[i], value);
        }
    }

    /**
     * @brief Generic CUDA kernel for in-place element-wise scalar operation.
     *
     * Modifies the input array in-place by applying a scalar value using
     * the provided operation functor.
     *
     * @tparam OpType Functor type (e.g. `ops::Addition`, `ops::Multiplication`, etc.)
     *
     * @param Xx    Input/Output array (f32x4 aligned)
     * @param value Scalar value to apply
     * @param N     Number of elements
     */
    template<typename OpType>
    __global__ void scalar_inplace(f32x4* __restrict__ Xx, const f32 value, const size_t N) {
        const OpType op{};

        const size_t vecN = N >> 2;
        const size_t tail_start = vecN << 2;

        CXM_CUDA_LOOP_1D(i, vecN) {
            const f32x4 x = Xx[i];
            Xx[i] = {
                op(x.x, value), op(x.y, value),
                op(x.z, value), op(x.w, value)
            };
        }

        f32* __restrict__ Xx_f = reinterpret_cast<f32*>(Xx);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xx_f[i] = op(Xx_f[i], value);
        }
    }

} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_SCALAR_CUH
