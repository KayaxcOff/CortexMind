//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_CAST_CUH
#define CORTEXMIND_CORE_ENGINE_CUDA_CAST_CUH

#include <CortexMind/core/Engine/CUDA/params.cuh>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief   Reinterprets a float* pointer as float4* (vectorized view)
     * @param   ptr   Pointer to float array (must be 16-byte aligned)
     * @return  Pointer to the same memory interpreted as float4 array
     *
     * @note    Useful for 128-bit vector loads/stores (coalesced access)
     * @note    Caller must ensure ptr is aligned to 16 bytes
     */
    [[nodiscard]] __device__ __host__
    inline f4x32* to_vec(f32* ptr) {
        return reinterpret_cast<f4x32*>(ptr);
    }
    /**
     * @brief   Const overload: float* → const float4*
     */
    [[nodiscard]] __device__ __host__
    inline const f4x32* to_vec(const f32* ptr) {
        return reinterpret_cast<const f4x32*>(ptr);
    }
    /**
     * @brief   Reinterprets a float4* pointer back to float*
     * @param   ptr   Pointer to float4 array
     * @return  Pointer to the same memory interpreted as float array
     */
    [[nodiscard]] __device__ __host__
    inline f32* to_scalar(f4x32* ptr) {
        return reinterpret_cast<f32*>(ptr);
    }
    /**
     * @brief   Const overload: const float4* → const float*
     */
    [[nodiscard]] __device__ __host__
    inline const f32* to_scalar(const f4x32* ptr) {
        return reinterpret_cast<const f32*>(ptr);
    }
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_CAST_CUH