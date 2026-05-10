//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_MATRIX_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_MATRIX_CUH

#include <CortexMind/framework/Engine/CUDA/types.cuh>
#include <CortexMind/framework/Tools/loop.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief Generic element-wise binary operation kernel (out-of-place).
     *
     * Applies a binary operation (`OpType`) between two arrays element by element.
     *
     * @tparam OpType Functor type (e.g. `ops::Addition`, `ops::Multiplication`, etc.)
     *
     * @param Xx Input array A
     * @param Xy Input array B
     * @param Xz Output array C = op(A, B)
     * @param N  Number of elements
     *
     * @note Uses `f32x4` vectorization for main loop + scalar tail for remainder.
     */
    template<typename OpType>
    __global__ void matrix(const f32x4* __restrict__ Xx, const f32x4* __restrict__ Xy, f32x4* __restrict__ Xz, const size_t N) {
        const OpType op{};

        const size_t vecN = N >> 2;
        const size_t tail_start = vecN << 2;

        CXM_CUDA_LOOP_1D(i, vecN) {
            const f32x4 x = Xx[i];
            const f32x4 y = Xy[i];
            Xz[i] = {
                op(x.x, y.x), op(x.y, y.y),
                op(x.z, y.z), op(x.w, y.w)
            };
        }

        const f32* __restrict__ Xx_f = reinterpret_cast<const f32*>(Xx);
        const f32* __restrict__ Xy_f = reinterpret_cast<const f32*>(Xy);
        f32* __restrict__       Xz_f = reinterpret_cast<f32*>(Xz);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xz_f[i] = op(Xx_f[i], Xy_f[i]);
        }
    }

    /**
     * @brief Generic element-wise binary operation kernel (in-place).
     *
     * Modifies the first array in-place: `X = op(X, Y)`
     *
     * @tparam OpType Functor type (e.g. `ops::Addition`, `ops::Multiplication`, etc.)
     *
     * @param Xx Input/Output array (will be modified)
     * @param Xy Input array B
     * @param N  Number of elements
     */
    template<typename OpType>
    __global__ void matrix_inplace(f32x4* __restrict__ Xx, const f32x4* __restrict__ Xy, const size_t N) {
        const OpType op{};

        const size_t vecN = N >> 2;
        const size_t tail_start = vecN << 2;

        CXM_CUDA_LOOP_1D(i, vecN) {
            const f32x4 x = Xx[i];
            const f32x4 y = Xy[i];
            Xx[i] = {
                op(x.x, y.x), op(x.y, y.y),
                op(x.z, y.z), op(x.w, y.w)
            };
        }

        f32* __restrict__ Xx_f = reinterpret_cast<f32*>(Xx);
        const f32* __restrict__ Xy_f = reinterpret_cast<const f32*>(Xy);

        CXM_CUDA_LOOP_TAIL(i, tail_start, N) {
            Xx_f[i] = op(Xx_f[i], Xy_f[i]);
        }
    }

} //namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_KERNELS_MATRIX_CUH