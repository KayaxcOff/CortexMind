//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_H
#define CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace cortex::_fw::cuda {
    using f4x32     = float4;
    using f2x64     = double2;
    using f16       = half;
    using bf16      = __nv_bfloat16;
    using bf2x16    = __nv_bfloat162;
    using i2x32     = int2;
    using i3x32     = int3;
    using i4x32     = int4;
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_H