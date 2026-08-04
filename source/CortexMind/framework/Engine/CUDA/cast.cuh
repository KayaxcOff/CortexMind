//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_CAST_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_CAST_CUH

#include <tlx/types.hpp>
#include <cuda_fp16.h>

namespace cortex::_fw::nv {
    template<typename T1, typename T2>
    T1* convert(T2*) = delete;

    template<>
    [[nodiscard]]
    inline __nv_bfloat16* convert(tlx::bfloat16* x) {
        return reinterpret_cast<__nv_bfloat16 *>(x);
    }
    template<>
    [[nodiscard]]
    inline const __nv_bfloat16* convert(tlx::bfloat16* x) {
        return reinterpret_cast<const __nv_bfloat16 *>(x);
    }
    template<>
    [[nodiscard]]
    inline const __nv_bfloat16* convert(const tlx::bfloat16* x) {
        return reinterpret_cast<const __nv_bfloat16 *>(x);
    }

    template<>
    [[nodiscard]]
    inline tlx::bfloat16* convert(__nv_bfloat16* x) {
        return reinterpret_cast<tlx::bfloat16 *>(x);
    }
    template<>
    [[nodiscard]]
    inline const tlx::bfloat16* convert(__nv_bfloat16* x) {
        return reinterpret_cast<const tlx::bfloat16 *>(x);
    }
    [[nodiscard]]
    inline const tlx::bfloat16* convert(const __nv_bfloat16* x) {
        return reinterpret_cast<const tlx::bfloat16 *>(x);
    }

    template<>
    [[nodiscard]]
    inline __nv_half* convert(tlx::half* x) {
        return reinterpret_cast<__nv_half *>(x);
    }
    template<>
    [[nodiscard]]
    inline const __nv_half* convert(tlx::half* x) {
        return reinterpret_cast<const __nv_half *>(x);
    }
    template<>
    [[nodiscard]]
    inline const __nv_half* convert(const tlx::half* x) {
        return reinterpret_cast<const __nv_half *>(x);
    }

    template<>
    [[nodiscard]]
    inline tlx::half* convert(__nv_half* x) {
        return reinterpret_cast<tlx::half *>(x);
    }
    template<>
    [[nodiscard]]
    inline const tlx::half* convert(__nv_half* x) {
        return reinterpret_cast<const tlx::half *>(x);
    }
    template<>
    [[nodiscard]]
    inline const tlx::half* convert(const __nv_half* x) {
        return reinterpret_cast<const tlx::half *>(x);
    }
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_CAST_CUH