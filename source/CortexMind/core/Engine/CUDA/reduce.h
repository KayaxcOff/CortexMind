//
// Created by muham on 31.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H
#define CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct Reduce {
        [[nodiscard]]
        static f32 sum(const bf16* Xx, size_t N);
        [[nodiscard]]
        static f32 mean(const bf16* Xx, size_t N);
        [[nodiscard]]
        static f32 variance(const bf16* Xx, size_t N, bool unbiased = true);
        [[nodiscard]]
        static f32 max(const bf16* Xx, size_t N);
        [[nodiscard]]
        static f32 min(const bf16* Xx, size_t N);
    };

} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H