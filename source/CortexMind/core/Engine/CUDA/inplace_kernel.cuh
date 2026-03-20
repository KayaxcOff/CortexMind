//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_INPLACE_KERNEL_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_INPLACE_KERNEL_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>

namespace cortex::_fw::cuda::kernels {
    /**
     * @brief   In-place element-wise binary operation: X[i] op= Y[i]
     * @tparam  Op      Functor type (e.g. op::Add, op::Mul, op::Div)
     * @param   Xx      Input/output array (float4 view, modified in-place)
     * @param   Xy      Second input array (const float4 view)
     * @param   idx     Total number of float elements (not float4 count)
     *
     * @note    Main loop processes 4 elements per iteration
     * @note    Tail (remaining 0–3 elements) handled via scalar path
     * @note    Op must be `__device__` callable with signature f32(f32, f32)
     * @note    Both pointers must be 16-byte aligned
     */
    template<typename Op>
    __global__ void inplace_kernel(f4x32* __restrict Xx, const f4x32* __restrict Xy, size_t idx) {
        Op op;
        CXM_CUDA_LOOP(i, idx / 4) {
            Xx[i] = {op(Xx[i].x, Xy[i].x), op(Xx[i].y, Xy[i].y),
                     op(Xx[i].z, Xy[i].z), op(Xx[i].w, Xy[i].w)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) {
            reinterpret_cast<f32*>(Xx)[base + tid] =
                op(reinterpret_cast<f32*>(Xx)[base + tid],
                   reinterpret_cast<const f32*>(Xy)[base + tid]);
        }
    }
    /**
     * @brief   In-place element-wise scalar operation: X[i] op= value
     * @tparam  Op      Functor type (e.g. op::Mul, op::Add, op::Div)
     * @param   Xx      Input/output array (float4 view, modified in-place)
     * @param   value   Scalar operand
     * @param   idx     Total number of float elements
     *
     * @note    Main loop processes 4 elements per iteration
     * @note    Tail handled via scalar path
     * @note    Op must be `__device__` callable with signature f32(f32, f32)
     * @note    Pointer must be 16-byte aligned
     */
    template<typename Op>
    __global__ void inplace_scalar_kernel(f4x32* __restrict Xx, f32 value, size_t idx) {
        Op op;
        CXM_CUDA_LOOP(i, idx / 4) {
            Xx[i] = {op(Xx[i].x, value), op(Xx[i].y, value),
                     op(Xx[i].z, value), op(Xx[i].w, value)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();
        if (tid < tail) {
            reinterpret_cast<f32*>(Xx)[base + tid] =
                op(reinterpret_cast<f32*>(Xx)[base + tid], value);
        }
    }

} // namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_INPLACE_KERNEL_CUH