//
// Created by muham on 17.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_BROADCAST_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_BROADCAST_CUH

#include <CortexMind/core/Tools/params.hpp>
#include <vector>

namespace cortex::_fw::cuda {
    /**
     * @brief   Launches broadcast kernels for element-wise operations with broadcasting
     *
     * Each method applies the corresponding binary operation with broadcasting rules:
     *   - Dimensions of size 1 are stretched to match the other tensor
     *   - Output shape is the broadcasted union (max per dimension)
     *
     * Typical usage:
     * @code
     *     std::vector<i64> shape_a = {2, 3, 4};
     *     std::vector<i64> shape_b = {1, 3, 1};
     *     std::vector<i64> shape_out = broadcast_shape(shape_a, shape_b);
     *     std::vector<i64> stride_a = compute_stride(shape_a);
     *     std::vector<i64> stride_b = compute_stride(shape_b);
     *
     *     cuda::broadcast_t::mul(d_a, d_b, d_out, shape_a, stride_a, shape_b, stride_b, shape_out);
     * @endcode
     */
    struct broadcast_t {
        /**
         * @brief   Broadcasted element-wise addition: Z[i] = X[i] + Y[i]
         * @param   Xx          Flattened input tensor A (device pointer)
         * @param   Xy          Flattened input tensor B (device pointer)
         * @param   Xz          Flattened output tensor (pre-allocated device pointer)
         * @param   shape_x     Shape of tensor A (host vector)
         * @param   stride_x    Strides of tensor A (host vector)
         * @param   shape_y     Shape of tensor B
         * @param   stride_y    Strides of tensor B
         * @param   shape_out   Broadcasted output shape (must match pre-allocated Xz)
         *
         * @pre     shape_out is the broadcasted shape of shape_x and shape_y
         * @pre     Xz has enough space (product of shape_out)
         * @note    Uses op::Add functor internally
         */
        static void add(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                        const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        const std::vector<i64>& shape_out);
        /**
         * @brief   Broadcasted element-wise addition: Z[i] = X[i] - Y[i]
         * @param   Xx          Flattened input tensor A (device pointer)
         * @param   Xy          Flattened input tensor B (device pointer)
         * @param   Xz          Flattened output tensor (pre-allocated device pointer)
         * @param   shape_x     Shape of tensor A (host vector)
         * @param   stride_x    Strides of tensor A (host vector)
         * @param   shape_y     Shape of tensor B
         * @param   stride_y    Strides of tensor B
         * @param   shape_out   Broadcasted output shape (must match pre-allocated Xz)
         *
         * @pre     shape_out is the broadcasted shape of shape_x and shape_y
         * @pre     Xz has enough space (product of shape_out)
         * @note    Uses op::Add functor internally
         */
        static void sub(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                        const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        const std::vector<i64>& shape_out);
        /**
         * @brief   Broadcasted element-wise addition: Z[i] = X[i] * Y[i]
         * @param   Xx          Flattened input tensor A (device pointer)
         * @param   Xy          Flattened input tensor B (device pointer)
         * @param   Xz          Flattened output tensor (pre-allocated device pointer)
         * @param   shape_x     Shape of tensor A (host vector)
         * @param   stride_x    Strides of tensor A (host vector)
         * @param   shape_y     Shape of tensor B
         * @param   stride_y    Strides of tensor B
         * @param   shape_out   Broadcasted output shape (must match pre-allocated Xz)
         *
         * @pre     shape_out is the broadcasted shape of shape_x and shape_y
         * @pre     Xz has enough space (product of shape_out)
         * @note    Uses op::Add functor internally
         */
        static void mul(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                        const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        const std::vector<i64>& shape_out);
        /**
         * @brief   Broadcasted element-wise addition: Z[i] = X[i] / Y[i]
         * @param   Xx          Flattened input tensor A (device pointer)
         * @param   Xy          Flattened input tensor B (device pointer)
         * @param   Xz          Flattened output tensor (pre-allocated device pointer)
         * @param   shape_x     Shape of tensor A (host vector)
         * @param   stride_x    Strides of tensor A (host vector)
         * @param   shape_y     Shape of tensor B
         * @param   stride_y    Strides of tensor B
         * @param   shape_out   Broadcasted output shape (must match pre-allocated Xz)
         *
         * @pre     shape_out is the broadcasted shape of shape_x and shape_y
         * @pre     Xz has enough space (product of shape_out)
         * @note    Uses op::Add functor internally
         */
        static void div(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                        const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                        const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                        const std::vector<i64>& shape_out);

    private:
        /**
         * @brief   Internal launch helper – uploads shapes/strides and calls templated kernel
         * @tparam  Op      Binary functor (op::Add, op::Sub, etc.)
         * @note    Temporary device buffers are freed immediately after kernel launch
         * @note    Does not synchronize — caller can do cudaDeviceSynchronize() if needed
         */
        static void launch(const f32* __restrict__ Xx, const f32* __restrict__ Xy, f32* __restrict__ Xz,
                           const std::vector<i64>& shape_x, const std::vector<i64>& stride_x,
                           const std::vector<i64>& shape_y, const std::vector<i64>& stride_y,
                           const std::vector<i64>& shape_out, int op);
    };
} // namespace cortex::_fw::cuda

#endif // CORTEXMIND_CORE_ENGINE_CUDA_BROADCAST_CUH