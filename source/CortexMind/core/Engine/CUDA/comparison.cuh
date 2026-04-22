//
// Created by muham on 22.04.2026.
//

#ifndef CORTEXMIND_COMPARISON_CUH
#define CORTEXMIND_COMPARISON_CUH

#include <CortexMind/core/Engine/CUDA/params.h>

namespace cortex::_fw::cuda {
    struct comparison {

        __device__ __forceinline static float4 gt(const float4& a, const float4& b) {
            return make_float4(
                a.x > b.x ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.y > b.y ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.z > b.z ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.w > b.w ? __int_as_float(0xFFFFFFFF) : 0.0f
            );
        }

        __device__ __forceinline static float4 lt(const float4& a, const float4& b) {
            return make_float4(
                a.x < b.x ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.y < b.y ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.z < b.z ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.w < b.w ? __int_as_float(0xFFFFFFFF) : 0.0f
            );
        }

        __device__ __forceinline static float4 eq(const float4& a, const float4& b) {
            return make_float4(
                a.x == b.x ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.y == b.y ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.z == b.z ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.w == b.w ? __int_as_float(0xFFFFFFFF) : 0.0f
            );
        }

        __device__ __forceinline static float4 ge(const float4& a, const float4& b) {
            return make_float4(
                a.x >= b.x ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.y >= b.y ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.z >= b.z ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.w >= b.w ? __int_as_float(0xFFFFFFFF) : 0.0f
            );
        }

        __device__ __forceinline static float4 le(const float4& a, const float4& b) {
            return make_float4(
                a.x <= b.x ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.y <= b.y ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.z <= b.z ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.w <= b.w ? __int_as_float(0xFFFFFFFF) : 0.0f
            );
        }

        __device__ __forceinline static float4 neq(const float4& a, const float4& b) {
            return make_float4(
                a.x != b.x ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.y != b.y ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.z != b.z ? __int_as_float(0xFFFFFFFF) : 0.0f,
                a.w != b.w ? __int_as_float(0xFFFFFFFF) : 0.0f
            );
        }

        __device__ __forceinline static i32 mask(const float4& x) {
            int m = 0;
            if (__float_as_int(x.x)) m |= 1 << 0;
            if (__float_as_int(x.y)) m |= 1 << 1;
            if (__float_as_int(x.z)) m |= 1 << 2;
            if (__float_as_int(x.w)) m |= 1 << 3;
            return m;
        }

        __device__ __forceinline static bool any(const float4& x) {
            return mask(x) != 0;
        }

        __device__ __forceinline static bool all(const float4& x) {
            return mask(x) == 0xF;
        }
    };
} // namespace cortex::_fw::cuda

#endif //CORTEXMIND_COMPARISON_CUH