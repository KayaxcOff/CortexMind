//
// Created by muham on 28.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_BROADCAST_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_BROADCAST_HPP

#include <CortexMind/core/Engine/AVX2/functions.hpp>
#include <CortexMind/core/Engine/AVX2/mask.hpp>
#include <CortexMind/core/Tools/broadcast.hpp>
#include <functional>

namespace cortex::_fw::avx2 {

    struct MatrixBroadcast {
        template <typename OpVec, typename OpScalar>
        static void row_broadcast(const f32* Xx, const f32* Xy, f32* Xz, size_t M, size_t N, OpVec op_vec, OpScalar op_scalar) {
            for (size_t row = 0; row < M; ++row) {
                const f32* x_row = Xx + row * N;
                f32* z_row       = Xz + row * N;

                size_t i = 0;
                for (; i + 8 <= N; i += 8) {
                    vec8f xv = loadu(x_row + i);
                    vec8f yv = loadu(Xy + i);
                    storeu(z_row + i, op_vec(xv, yv));
                }

                if (i < N) {
                    const mask m(N - i);
                    vec8f xv = m.load(x_row + i);
                    vec8f yv = m.load(Xy + i);
                    m.store(z_row + i, op_vec(xv, yv));
                }
            }
        }

        template <typename OpVec, typename OpScalar>
        static void col_broadcast(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N, OpVec op_vec, OpScalar op_scalar) {
            for (size_t row = 0; row < M; ++row) {
                const f32* x_row = Xx + row * N;
                f32* z_row       = Xz + row * N;

                vec8f y_broadcast = set1(Xy[row]);

                size_t i = 0;
                for (; i + 8 <= N; i += 8) {
                    vec8f xv = loadu(x_row + i);
                    storeu(z_row + i, op_vec(xv, y_broadcast));
                }

                if (i < N) {
                    const mask m(N - i);
                    vec8f xv = m.load(x_row + i);
                    m.store(z_row + i, op_vec(xv, y_broadcast));
                }
            }
        }

        template <typename OpScalar>
        static void generic_broadcast(const f32* Xx, const f32* Xy, f32* Xz, const BroadcastInfo& info, OpScalar op_scalar) {
            size_t numel = 1;
            for (i32 d = 0; d < info.ndim; ++d) {
                numel *= info.shape[d];
            }

            for (size_t i = 0; i < numel; ++i) {
                size_t offset_x = 0;
                size_t offset_y = 0;
                size_t offset_z = 0;

                size_t linear_idx = i;
                for (i32 d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = linear_idx % info.shape[d];
                    offset_x += coord * info.stride_x[d];
                    offset_y += coord * info.stride_y[d];
                    offset_z += coord * info.stride_z[d];
                    linear_idx /= info.shape[d];
                }

                Xz[offset_z] = op_scalar(Xx[offset_x], Xy[offset_y]);
            }
        }
    };
} // namespace cortex::_fw::avx2
#endif //CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_BROADCAST_HPP