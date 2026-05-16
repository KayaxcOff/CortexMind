//
// Created by muham on 8.05.2026.
//

//
// Created by muham on 8.05.2026.
//

#include "CortexMind/framework/Engine/AVX2/broadcast.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <vector>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

namespace {
    template<typename OpVec, typename OpScalar>
    __forceinline void apply_row(const f32* __restrict x_row, const f32* __restrict y_row, f32* __restrict z_row, const size_t N, OpVec op_vec, OpScalar op_scalar) {
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            storeu(z_row + i, op_vec(loadu(x_row + i), loadu(y_row + i)));
        }
        for (; i < N; ++i) {
            z_row[i] = op_scalar(x_row[i], y_row[i]);
        }
    }

    template<typename OpVec, typename OpScalar>
    __forceinline void apply_row_ip(f32* __restrict x_row, const f32* __restrict y_row, const size_t N, OpVec op_vec, OpScalar op_scalar) {
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            storeu(x_row + i, op_vec(loadu(x_row + i), loadu(y_row + i)));
        }
        for (; i < N; ++i) {
            x_row[i] = op_scalar(x_row[i], y_row[i]);
        }
    }

    template<typename OpVec, typename OpScalar>
    __forceinline void apply_row_broadcast(const f32* __restrict x_row, const vec8f yv, const f32 ys, f32* __restrict z_row, const size_t N, OpVec op_vec, OpScalar op_scalar) {
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            storeu(z_row + i, op_vec(loadu(x_row + i), yv));
        }
        for (; i < N; ++i) {
            z_row[i] = op_scalar(x_row[i], ys);
        }
    }

    template<typename OpVec, typename OpScalar>
    __forceinline void apply_row_broadcast_ip(f32* __restrict x_row, const vec8f yv, const f32 ys, const size_t N, OpVec op_vec, OpScalar op_scalar) {
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            storeu(x_row + i, op_vec(loadu(x_row + i), yv));
        }
        for (; i < N; ++i) {
            x_row[i] = op_scalar(x_row[i], ys);
        }
    }

    template<typename OpVec, typename OpScalar>
    void execute_broadcast_2d(
    const f32* __restrict x, const std::vector<int64_t>& x_shape, const std::vector<int64_t>& x_strides, const f32* __restrict y, const std::vector<int64_t>& y_shape, const std::vector<int64_t>& y_strides, f32* __restrict z, OpVec op_vec, OpScalar op_scalar) {
        const size_t rows = x_shape[0];
        const size_t cols = x_shape[1];

        const int64_t y_stride_row = y_strides[0];
        const int64_t y_stride_col = y_strides[1];

        for (size_t r = 0; r < rows; ++r) {
            const f32* x_row = x + r * x_strides[0];
            f32* z_row = z + r * cols;

            const f32* y_row = y + r * y_stride_row;

            if (y_stride_col == 0) {
                const f32 ys = y_row[0];
                const vec8f yv = _mm256_set1_ps(ys);
                apply_row_broadcast(x_row, yv, ys, z_row, cols, op_vec, op_scalar);
            } else {
                apply_row(x_row, y_row, z_row, cols, op_vec, op_scalar);
            }
        }
    }

} //unnamed namespace

void Broadcast::row_add(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row(Xx + row * N, Xy, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::add(a, b); },
            [](const f32 a, const f32 b)     { return a + b; });
    }
}

void Broadcast::row_sub(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row(Xx + row * N, Xy, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::sub(a, b); },
            [](const f32 a, const f32 b)     { return a - b; });
    }
}

void Broadcast::row_mul(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row(Xx + row * N, Xy, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::mul(a, b); },
            [](const f32 a, const f32 b)     { return a * b; });
    }
}

void Broadcast::row_div(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row(Xx + row * N, Xy, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::div(a, b); },
            [](const f32 a, const f32 b)     { return a / b; });
    }
}

// ------------------------------------------------------------------ //
//  Row broadcast — in-place                                            //
// ------------------------------------------------------------------ //

void Broadcast::row_add(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row_ip(Xx + row * N, Xy, N,
            [](const vec8f a, const vec8f b) { return avx2::add(a, b); },
            [](const f32 a, const f32 b)     { return a + b; });
    }
}

void Broadcast::row_sub(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row_ip(Xx + row * N, Xy, N,
            [](const vec8f a, const vec8f b) { return avx2::sub(a, b); },
            [](const f32 a, const f32 b)     { return a - b; });
    }
}

void Broadcast::row_mul(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row_ip(Xx + row * N, Xy, N,
            [](const vec8f a, const vec8f b) { return avx2::mul(a, b); },
            [](const f32 a, const f32 b)     { return a * b; });
    }
}

void Broadcast::row_div(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        apply_row_ip(Xx + row * N, Xy, N,
            [](const vec8f a, const vec8f b) { return avx2::div(a, b); },
            [](const f32 a, const f32 b)     { return a / b; });
    }
}

// ------------------------------------------------------------------ //
//  Col broadcast — out-of-place                                        //
// ------------------------------------------------------------------ //

void Broadcast::col_add(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast(Xx + row * N, set1(ys), ys, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::add(a, b); },
            [](const f32 a, const f32 b)     { return a + b; });
    }
}

void Broadcast::col_sub(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast(Xx + row * N, set1(ys), ys, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::sub(a, b); },
            [](const f32 a, const f32 b)     { return a - b; });
    }
}

void Broadcast::col_mul(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast(Xx + row * N, set1(ys), ys, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::mul(a, b); },
            [](const f32 a, const f32 b)     { return a * b; });
    }
}

void Broadcast::col_div(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast(Xx + row * N, set1(ys), ys, Xz + row * N, N,
            [](const vec8f a, const vec8f b) { return avx2::div(a, b); },
            [](const f32 a, const f32 b)     { return a / b; });
    }
}

// ------------------------------------------------------------------ //
//  Col broadcast — in-place                                            //
// ------------------------------------------------------------------ //

void Broadcast::col_add(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast_ip(Xx + row * N, set1(ys), ys, N,
            [](const vec8f a, const vec8f b) { return avx2::add(a, b); },
            [](const f32 a, const f32 b)     { return a + b; });
    }
}

void Broadcast::col_sub(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast_ip(Xx + row * N, set1(ys), ys, N,
            [](const vec8f a, const vec8f b) { return avx2::sub(a, b); },
            [](const f32 a, const f32 b)     { return a - b; });
    }
}

void Broadcast::col_mul(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast_ip(Xx + row * N, set1(ys), ys, N,
            [](const vec8f a, const vec8f b) { return avx2::mul(a, b); },
            [](const f32 a, const f32 b)     { return a * b; });
    }
}

void Broadcast::col_div(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    for (size_t row = 0; row < M; ++row) {
        const f32 ys = Xy[row];
        apply_row_broadcast_ip(Xx + row * N, set1(ys), ys, N,
            [](const vec8f a, const vec8f b) { return avx2::div(a, b); },
            [](const f32 a, const f32 b)     { return a / b; });
    }
}