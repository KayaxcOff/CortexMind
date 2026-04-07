//
// Created by muham on 8.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_H
#define CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace cortex::_fw::cuda {
    using f32x4     = float4;
    using f64x2     = double2;
    using f16       = half;
    using bf16      = __nv_bfloat16;
    using bf16x2    = __nv_bfloat162;
    using i32x2     = int2;
    using i32x3     = int3;
    using i32x4     = int4;
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_H