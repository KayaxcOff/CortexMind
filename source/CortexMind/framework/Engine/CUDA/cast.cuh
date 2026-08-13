//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_CAST_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_CAST_CUH

#include <tlx/types.hpp>
#include <cuda_fp16.h>

namespace cortex::_fw::nv {
    [[nodiscard]]
    inline float4* convert(float* x) {
        return reinterpret_cast<float4 *>(x);
    }
    [[nodiscard]]
    inline const float4* convert(const float* x) {
        return reinterpret_cast<const float4 *>(x);
    }

    [[nodiscard]]
    inline float* convert(float4* x) {
        return reinterpret_cast<float *>(x);
    }
    [[nodiscard]]
    inline const float* convert(const float4* x) {
        return reinterpret_cast<const float *>(x);
    }

    [[nodiscard]]
    inline __nv_bfloat162* convert(tlx::bfloat16* x) {
        return reinterpret_cast<__nv_bfloat162 *>(x);
    }
    [[nodiscard]]
    inline const __nv_bfloat162* convert(const tlx::bfloat16* x) {
        return reinterpret_cast<const __nv_bfloat162 *>(x);
    }

    [[nodiscard]]
    inline tlx::bfloat16* convert(__nv_bfloat16* x) {
        return reinterpret_cast<tlx::bfloat16 *>(x);
    }
    [[nodiscard]]
    inline const tlx::bfloat16* convert(const __nv_bfloat16* x) {
        return reinterpret_cast<const tlx::bfloat16 *>(x);
    }

    [[nodiscard]]
    inline tlx::bfloat16* convert(__nv_bfloat162* x) {
        return reinterpret_cast<tlx::bfloat16 *>(x);
    }
    [[nodiscard]]
    inline const tlx::bfloat16* convert(const __nv_bfloat162* x) {
        return reinterpret_cast<const tlx::bfloat16 *>(x);
    }

    [[nodiscard]]
    inline __nv_half2* convert(tlx::half* x) {
        return reinterpret_cast<__nv_half2 *>(x);
    }
    [[nodiscard]]
    inline const __nv_half2* convert(const tlx::half* x) {
        return reinterpret_cast<const __nv_half2 *>(x);
    }

    [[nodiscard]]
    inline tlx::half* convert(__nv_half* x) {
        return reinterpret_cast<tlx::half *>(x);
    }
    [[nodiscard]]
    inline const tlx::half* convert(const __nv_half* x) {
        return reinterpret_cast<const tlx::half *>(x);
    }

    [[nodiscard]]
    inline tlx::half* convert(__nv_half2* x) {
        return reinterpret_cast<tlx::half *>(x);
    }
    [[nodiscard]]
    inline const tlx::half* convert(const __nv_half2* x) {
        return reinterpret_cast<const tlx::half *>(x);
    }
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_CAST_CUH