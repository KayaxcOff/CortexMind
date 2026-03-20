//
// Created by muham on 17.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_BROADCAST_KERNEL_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_BROADCAST_KERNEL_CUH

#include <CortexMind/core/Engine/CUDA/op.cuh>
#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief   Generic element-wise broadcast operation: Z[i] = op(X[i], Y[i])
     * @tparam  Op      Binary functor (e.g. op::Add, op::Mul, op::Div)
     * @param   Xx          First input tensor (flattened device pointer)
     * @param   Xy          Second input tensor (flattened device pointer)
     * @param   Xz          Output tensor (pre-allocated, flattened device pointer)
     * @param   shape_x     Device pointer to shape of X (ndim elements)
     * @param   stride_x    Device pointer to strides of X (ndim elements)
     * @param   shape_y     Device pointer to shape of Y
     * @param   stride_y    Device pointer to strides of Y
     * @param   shape_out   Device pointer to broadcasted output shape
     * @param   ndim        Number of dimensions (rank of tensors)
     * @param   numel       Total number of output elements (product of shape_out)
     *
     * @note    Kernel uses linear output index → computes multi-dim coordinates
     * @note    When dim=1, input index fixed at 0 (broadcasting rule)
     * @note    No bounds checking inside loop → caller must guarantee correct shapes/strides
     * @note    Launch recommendation:
     *          dim3 grid = grid1d((numel + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
     *          broadcast_kernel<op::Mul><<<grid, BLOCK_SIZE_1D>>>(...)
     * @note    For small tensors CPU may be faster
     */
    template<typename Op>
    __global__ void broadcast_kernel(
        const f32* __restrict__ Xx,
        const f32* __restrict__ Xy,
        f32* __restrict__ Xz,
        const i64* __restrict__ shape_x,
        const i64* __restrict__ stride_x,
        const i64* __restrict__ shape_y,
        const i64* __restrict__ stride_y,
        const i64* __restrict__ shape_out,
        i64 ndim,
        size_t numel)
    {
        Op op;
        CXM_CUDA_LOOP(out_idx, numel) {
            // linear → multi-dim (output shape'e göre)
            size_t remaining = out_idx;
            i64 idx_x = 0;
            i64 idx_y = 0;

            for (i64 d = ndim - 1; d >= 0; --d) {
                const i64 coord = static_cast<i64>(remaining) % shape_out[d];
                remaining /= static_cast<size_t>(shape_out[d]);

                // boyut 1 ise indeks 0, değilse coord
                idx_x += (shape_x[d] == 1 ? 0 : coord) * stride_x[d];
                idx_y += (shape_y[d] == 1 ? 0 : coord) * stride_y[d];
            }

            Xz[out_idx] = op(Xx[idx_x], Xy[idx_y]);
        }
    }

} // namespace cortex::_fw::cuda::kernels

#endif // CORTEXMIND_CORE_ENGINE_CUDA_BROADCAST_KERNEL_CUH