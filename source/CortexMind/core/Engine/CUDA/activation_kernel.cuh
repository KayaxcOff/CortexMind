//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_KERNEL_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_KERNEL_CUH

// Because Swish and LeakyRelu require more variables as
// input compared to other activation functions, we moved
// them to separate kernels to avoid voting issues.

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief   Generic element-wise unary activation: Z[i] = op(X[i])
     * @tparam  Op      Functor type (e.g. op::Relu, op::Gelu, op::Sigmoid)
     * @param   Xx      Input/output array (in-place) as float4 view
     * @param   idx     Total number of float elements (not float4 count)
     *
     * @note    Main loop processes 4 elements per iteration
     * @note    Tail (remaining 0–3 elements) handled via scalar path
     * @note    In-place operation (Xx is both read and written)
     * @note    Op must be `__device__` callable with signature f32(f32)
     */
    template<typename Op>
    __global__ void activation_kernel(f4x32* __restrict Xx, size_t idx) {
        Op op;
        CXM_CUDA_LOOP(i, idx / 4) {
            Xx[i] = {op(Xx[i].x), op(Xx[i].y), op(Xx[i].z), op(Xx[i].w)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) {
            reinterpret_cast<f32*>(Xx)[base + tid] =
                op(reinterpret_cast<f32*>(Xx)[base + tid]);
        }
    }
    /**
     * @brief   Leaky ReLU: Z[i] = x > 0 ? x : alpha * x    (in-place)
     * @param   Xx      Input/output array (float4 view)
     * @param   alpha   Negative slope (typical: 0.01)
     * @param   idx     Number of float elements
     *
     * @note    Vectorized main loop + scalar tail
     * @note    In-place operation
     */
    __global__ void leaky_relu_kernel(f4x32* __restrict Xx, f32 alpha, size_t idx) {
        CXM_CUDA_LOOP(i, idx / 4) {
            Xx[i] = {Xx[i].x > 0.0f ? Xx[i].x : Xx[i].x * alpha,
                     Xx[i].y > 0.0f ? Xx[i].y : Xx[i].y * alpha,
                     Xx[i].z > 0.0f ? Xx[i].z : Xx[i].z * alpha,
                     Xx[i].w > 0.0f ? Xx[i].w : Xx[i].w * alpha};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) {
            f32& val = reinterpret_cast<f32*>(Xx)[base + tid];
            val = val > 0.0f ? val : val * alpha;
        }
    }
    /**
     * @brief   Swish (β-parameterized): Z[i] = x × sigmoid(β x)    (in-place)
     * @param   Xx      Input/output array (float4 view)
     * @param   beta    Scaling factor (typical: 1.0)
     * @param   idx     Number of float elements
     *
     * @note    Uses expf — may benefit from -use_fast_math
     * @note    In-place operation
     */
    __global__ void swish_kernel(f4x32* __restrict Xx, f32 beta, size_t idx) {
        CXM_CUDA_LOOP(i, idx / 4) {
            Xx[i] = {Xx[i].x * (1.0f / (1.0f + expf(-beta * Xx[i].x))),
                     Xx[i].y * (1.0f / (1.0f + expf(-beta * Xx[i].y))),
                     Xx[i].z * (1.0f / (1.0f + expf(-beta * Xx[i].z))),
                     Xx[i].w * (1.0f / (1.0f + expf(-beta * Xx[i].w)))};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) {
            f32& val = reinterpret_cast<f32*>(Xx)[base + tid];
            val = val * (1.0f / (1.0f + expf(-beta * val)));
        }
    }

} // namespace cortex::_fw::cuda::kernels

#endif // CORTEXMIND_CORE_ENGINE_CUDA_ACTIVATION_KERNEL_CUH
