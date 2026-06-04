//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_GENERAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_GENERAL_HPP

#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief Generic vectorized N-D broadcast operation.
     *
     * Strategy:
     * 1. Check if innermost dimension is contiguous
     * 2. If yes: iterate outer dims, vectorize innermost
     * 3. If no: fallback to scalar loop
     */
    template<typename OpVec, typename OpScalar>
    void execute_broadcast_nd(const f32* __restrict x, const f32* __restrict y, f32* __restrict z, const BroadcastInfo& info, OpVec op_vec, OpScalar op_scalar) {
        const bool contiguous = (info.stride_x[info.ndim - 1] == 1 && info.stride_y[info.ndim - 1] == 1 && info.stride_z[info.ndim - 1] == 1);

        if (!contiguous) {
            size_t total = 1;
            for (i32 d = 0; d < info.ndim; ++d) total *= info.shape[d];

            for (size_t i = 0; i < total; ++i) {
                size_t ox = 0, oy = 0, oz = 0, idx = i;
                for (i32 d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = idx % info.shape[d];
                    ox += coord * info.stride_x[d];
                    oy += coord * info.stride_y[d];
                    oz += coord * info.stride_z[d];
                    idx /= info.shape[d];
                }
                z[oz] = op_scalar(x[ox], y[oy]);
            }
            return;
        }

        const size_t inner_size = info.shape[info.ndim - 1];
        std::vector<size_t> indices(info.ndim, 0);

        while (true) {
            size_t ox = 0, oy = 0, oz = 0;
            for (i32 d = 0; d < info.ndim - 1; ++d) {
                ox += indices[d] * info.stride_x[d];
                oy += indices[d] * info.stride_y[d];
                oz += indices[d] * info.stride_z[d];
            }

            size_t i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f xv = loadu(x + ox + i);
                const vec8f yv = loadu(y + oy + i);
                storeu(z + oz + i, op_vec(xv, yv));
            }

            for (; i < inner_size; ++i) {
                z[oz + i] = op_scalar(x[ox + i], y[oy + i]);
            }

            int d = info.ndim - 2;
            while (d >= 0) {
                indices[d]++;
                if (indices[d] < info.shape[d]) {
                    break;
                }
                indices[d] = 0;
                d--;
            }

            if (d < 0) {
                break;
            }
        }
    }

    inline void general_broadcast_add(const f32* __restrict x, const f32* __restrict y, f32* __restrict z, const BroadcastInfo& info) {
        execute_broadcast_nd(x, y, z, info,
            [](const vec8f a, const vec8f b) { return add(a, b); },
            [](const f32 a, const f32 b)     { return a + b; });
    }

    inline void general_broadcast_sub(const f32* __restrict x, const f32* __restrict y, f32* __restrict z, const BroadcastInfo& info) {
        execute_broadcast_nd(x, y, z, info,
            [](const vec8f a, const vec8f b) { return sub(a, b); },
            [](const f32 a, const f32 b)     { return a - b; });
    }

    inline void general_broadcast_mul(const f32* __restrict x, const f32* __restrict y, f32* __restrict z, const BroadcastInfo& info) {
        execute_broadcast_nd(x, y, z, info,
            [](const vec8f a, const vec8f b) { return mul(a, b); },
            [](const f32 a, const f32 b)     { return a * b; });
    }

    inline void general_broadcast_div(const f32* __restrict x, const f32* __restrict y, f32* __restrict z, const BroadcastInfo& info) {
        execute_broadcast_nd(x, y, z, info,
            [](const vec8f a, const vec8f b) { return div(a, b); },
            [](const f32 a, const f32 b)     { return a / b; });
    }
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_GENERAL_HPP