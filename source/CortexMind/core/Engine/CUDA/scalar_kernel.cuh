//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_KERNEL_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_KERNEL_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Engine/CUDA/utils.cuh>

namespace cortex::_fw::cuda::kernels {
    template<typename Op>
    __global__ void scalar_kernel(const f4x32* __restrict Xx, f32 value, f4x32* __restrict Xz, size_t idx) {
            Op op;
        CXM_CUDA_LOOP(i, idx / 4) {
            Xz[i] = {op(Xx[i].x, value), op(Xx[i].y, value),
                op(Xx[i].z, value), op(Xx[i].w, value)};
        }
        const size_t tail = idx & 3;
        const size_t base = idx - tail;
        const size_t tid  = global_thread_id();

        if (tid < tail) {
            reinterpret_cast<f32*>(Xz)[base + tid] = op(reinterpret_cast<const f32*>(Xx)[base + tid], value);
        }
    }
} // namespace cortex::_fw::cuda::kernels

#endif //CORTEXMIND_CORE_ENGINE_CUDA_SCALAR_KERNEL_CUH