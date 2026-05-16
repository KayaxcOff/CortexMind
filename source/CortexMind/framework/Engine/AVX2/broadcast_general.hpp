//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_GENERAL_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_GENERAL_HPP

#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <CortexMind/framework/Tools/err.hpp>

// avx2/broadcast_general.hpp — header-only, inline
namespace cortex::_fw::avx2 {

    inline void general_broadcast_add( const f32* __restrict x, const f32* __restrict y, f32* __restrict z, const BroadcastInfo&  info) {

        size_t total = 1;
        for (i32 d = 0; d < info.ndim; ++d) total *= info.shape[d];


        for (size_t i = 0; i < total; ++i) {
            size_t ox = 0, oy = 0, oz = 0, idx = i;

            for (i32 d = info.ndim - 1; d >= 0; --d) {
                const size_t coord = idx % info.shape[d];
                ox  += coord * info.stride_x[d];
                oy  += coord * info.stride_y[d];
                oz  += coord * info.stride_z[d];
                idx /= info.shape[d];
            }

            z[oz] = x[ox] + y[oy];
        }
    }

    inline void general_broadcast_sub(const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
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
            z[oz] = x[ox] - y[oy];
        }
    }

    inline void general_broadcast_mul(const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
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
            z[oz] = x[ox] * y[oy];
        }
    }

    inline void general_broadcast_div(const f32* x, const f32* y, f32* z, const BroadcastInfo& info) {
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
            z[oz] = x[ox] / y[oy];
        }
    }

} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_GENERAL_HPP