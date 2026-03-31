//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_UTILITIES_CUH
#define CORTEXMIND_CORE_TOOLS_UTILITIES_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    [[nodiscard]]
    __device__ inline size_t global_thread_id() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }
    [[nodiscard]]
    __host__ __device__ constexpr size_t round_up(const size_t n, const size_t align) {
        return (n + align - 1) / align * align;
    }
    [[nodiscard]]
    __device__ __host__ inline f4x32* to_vec(f32* ptr) {
        return reinterpret_cast<f4x32*>(ptr);
    }
    [[nodiscard]]
    __device__ __host__ inline const f4x32* to_vec(const f32* ptr) {
        return reinterpret_cast<const f4x32*>(ptr);
    }
    [[nodiscard]]
    __device__ __host__ inline f32* to_scalar(f4x32* ptr) {
        return reinterpret_cast<f32*>(ptr);
    }
    [[nodiscard]]
    __device__ __host__ inline const f32* to_scalar(const f4x32* ptr) {
        return reinterpret_cast<const f32*>(ptr);
    }
    [[nodiscard]]
    __device__ __host__ inline bf16 to_bf16(const f32 x) {
        return __float2bfloat16(x);
    }

    [[nodiscard]]
    __device__ __host__ inline f32 to_f32(const bf16 x) {
        return __bfloat162float(x);
    }

    [[nodiscard]]
    __device__ __host__ inline bf2x16 to_bf162(const f32 x, const f32 y) {
        return __floats2bfloat162_rn(x, y);
    }

    [[nodiscard]]
    __device__ __host__ inline f32 bf162_lo(const bf2x16 v) {
        return __bfloat162float(v.x);
    }
    [[nodiscard]]
    __device__ __host__ inline f32 bf162_hi(const bf2x16 v) {
        return __bfloat162float(v.y);
    }


    __device__ inline void f32x4_to_bf16x4(const f4x32 src, bf16* dst){
        dst[0] = __float2bfloat16(src.x);
        dst[1] = __float2bfloat16(src.y);
        dst[2] = __float2bfloat16(src.z);
        dst[3] = __float2bfloat16(src.w);
    }


    __device__ inline f4x32 bf16x4_to_f32x4(const bf16* src) {
        return make_float4(
            __bfloat162float(src[0]),
            __bfloat162float(src[1]),
            __bfloat162float(src[2]),
            __bfloat162float(src[3])
        );
    }
} //namespace cortex::_fw::cuda

#define CXM_CUDA_LOOP_1D(i, N) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (N); \
        i += blockDim.x * gridDim.x)

#endif //CORTEXMIND_CORE_TOOLS_UTILITIES_CUH