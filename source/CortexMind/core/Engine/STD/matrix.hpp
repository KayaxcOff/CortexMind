//
// Created by muham on 26.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP

#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/core/Tools/broadcast.hpp>

namespace cortex::_fw::stl {
    struct matrix {
        static void add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);
        static void div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t N);

        static void matmul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t xN, size_t yN, size_t zN);

        static void add(f32* Xx, const f32* __restrict Xy, size_t N);
        static void sub(f32* Xx, const f32* __restrict Xy, size_t N);
        static void mul(f32* Xx, const f32* __restrict Xy, size_t N);
        static void div(f32* Xx, const f32* __restrict Xy, size_t N);

        template<typename OpScalar>
        static void broadcast(const f32* Xx, const f32* Xy, f32* Xz, const BroadcastInfo& info, OpScalar op) {
            size_t numel = 1;
            for (int d = 0; d < info.ndim; ++d)
                numel *= info.shape[d];

            for (size_t i = 0; i < numel; ++i) {
                size_t offset_x  = 0;
                size_t offset_y  = 0;
                size_t offset_z  = 0;
                size_t linear_idx = i;

                for (int d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = linear_idx % info.shape[d];
                    offset_x  += coord * info.stride_x[d];
                    offset_y  += coord * info.stride_y[d];
                    offset_z  += coord * info.stride_z[d];
                    linear_idx /= info.shape[d];
                }

                Xz[offset_z] = op(Xx[offset_x], Xy[offset_y]);
            }
        }
    };
} //namespace cortex::_fw::stl

#endif //CORTEXMIND_CORE_ENGINE_STD_MATRIX_HPP