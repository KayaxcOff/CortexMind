//
// Created by muham on 9.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H
#define CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct ReduceOp {
        ReduceOp();

        [[nodiscard]]
        f32 sum(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 mean(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 var(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 std(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 min(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 max(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 norm1(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 norm2(const f32* __restrict x, size_t N);
        [[nodiscard]]
        f32 dot(const f32* __restrict Xx, const f32* __restrict Xy, size_t N);
    private:
        f32* host_output;
        f32* cuda_output;
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H