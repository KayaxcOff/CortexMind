//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_OPERATIONS_H
#define CORTEXMIND_CORE_TOOLS_OPERATIONS_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda::ops {
    struct Addition {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx + Xy;
        }
    };
    struct Subtraction {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx - Xy;
        }
    };
    struct Multiplication {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx * Xy;
        }
    };
    struct Division {
        __device__ f32 operator()(const f32 Xx, const f32 Xy) const {
            return Xx / Xy;
        }
    };
} //namespace cortex::_fw::cuda::ops

#endif //CORTEXMIND_CORE_TOOLS_OPERATIONS_H