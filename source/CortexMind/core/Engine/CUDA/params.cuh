//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cortex::_fw::cuda {
     using f4x32    = float4;
     using f2x64    = double2;
     using f16      = half;
     using i2x32    = int2;
     using i3x32    = int3;
     using i4x32    = int4;
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_PARAMS_CUH