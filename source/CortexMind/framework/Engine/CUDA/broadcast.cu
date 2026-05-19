//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Engine/CUDA/broadcast.cuh"
#include <CortexMind/framework/Engine/CUDA/Kernels/broadcast.cuh>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/kernel_operations.hpp>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

namespace {
    // row_broadcast grid: 2D, ceiling division her iki eksende
    inline dim3 row_grid(const size_t M, const size_t N) {
        return dim3(
            static_cast<unsigned>((N + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D),
            static_cast<unsigned>((M + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D)
        );
    }
    inline dim3 row_block() {
        return dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    }

    // col_broadcast grid: gridDim.y = M tam olarak
    inline dim3 col_grid(const size_t M, const size_t N) {
        return dim3(
            static_cast<unsigned>((N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D),
            static_cast<unsigned>(M)
        );
    }
    inline dim3 col_block() {
        return dim3(BLOCK_SIZE_1D, 1);
    }
} //unnamed namespace

// ------------------------------------------------------------------ //
//  Row broadcast — out-of-place                                        //
// ------------------------------------------------------------------ //

void Broadcast::row_add(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::row_broadcast<<<row_grid(M, N), row_block()>>>(Xx, Xy, Xz, M, N, ops::Addition{});
}
void Broadcast::row_sub(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::row_broadcast<<<row_grid(M, N), row_block()>>>(Xx, Xy, Xz, M, N, ops::Subtraction{});
}
void Broadcast::row_mul(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::row_broadcast<<<row_grid(M, N), row_block()>>>(Xx, Xy, Xz, M, N, ops::Multiplication{});
}
void Broadcast::row_div(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::row_broadcast<<<row_grid(M, N), row_block()>>>(Xx, Xy, Xz, M, N, ops::Division{});
}

// ------------------------------------------------------------------ //
//  Row broadcast — in-place                                            //
// ------------------------------------------------------------------ //

void Broadcast::row_add(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::row_broadcast_ip<<<row_grid(M, N), row_block()>>>(Xx, Xy, M, N, ops::Addition{});
}
void Broadcast::row_sub(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::row_broadcast_ip<<<row_grid(M, N), row_block()>>>(Xx, Xy, M, N, ops::Subtraction{});
}
void Broadcast::row_mul(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::row_broadcast_ip<<<row_grid(M, N), row_block()>>>(Xx, Xy, M, N, ops::Multiplication{});
}
void Broadcast::row_div(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::row_broadcast_ip<<<row_grid(M, N), row_block()>>>(Xx, Xy, M, N, ops::Division{});
}

// ------------------------------------------------------------------ //
//  Col broadcast — out-of-place                                        //
// ------------------------------------------------------------------ //

void Broadcast::col_add(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::col_broadcast<<<col_grid(M, N), col_block()>>>(Xx, Xy, Xz, M, N, ops::Addition{});
}
void Broadcast::col_sub(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::col_broadcast<<<col_grid(M, N), col_block()>>>(Xx, Xy, Xz, M, N, ops::Subtraction{});
}
void Broadcast::col_mul(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::col_broadcast<<<col_grid(M, N), col_block()>>>(Xx, Xy, Xz, M, N, ops::Multiplication{});
}
void Broadcast::col_div(const f32* Xx, const f32* Xy, f32* Xz, const size_t M, const size_t N) {
    kernels::col_broadcast<<<col_grid(M, N), col_block()>>>(Xx, Xy, Xz, M, N, ops::Division{});
}

// ------------------------------------------------------------------ //
//  Col broadcast — in-place                                            //
// ------------------------------------------------------------------ //

void Broadcast::col_add(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::col_broadcast_ip<<<col_grid(M, N), col_block()>>>(Xx, Xy, M, N, ops::Addition{});
}
void Broadcast::col_sub(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::col_broadcast_ip<<<col_grid(M, N), col_block()>>>(Xx, Xy, M, N, ops::Subtraction{});
}
void Broadcast::col_mul(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::col_broadcast_ip<<<col_grid(M, N), col_block()>>>(Xx, Xy, M, N, ops::Multiplication{});
}
void Broadcast::col_div(f32* Xx, const f32* Xy, const size_t M, const size_t N) {
    kernels::col_broadcast_ip<<<col_grid(M, N), col_block()>>>(Xx, Xy, M, N, ops::Division{});
}

void Broadcast::general_add(const f32* Xx, const f32* Xy, f32* Xz,
                              const BroadcastInfo& info, const size_t total) {
    kernels::general_broadcast<<<grid1d(total), BLOCK_SIZE_1D>>>(
        Xx, Xy, Xz, info, total, ops::Addition{});
}

void Broadcast::general_sub(const f32* Xx, const f32* Xy, f32* Xz,
                              const BroadcastInfo& info, const size_t total) {
    kernels::general_broadcast<<<grid1d(total), BLOCK_SIZE_1D>>>(
        Xx, Xy, Xz, info, total, ops::Subtraction{});
}

void Broadcast::general_mul(const f32* Xx, const f32* Xy, f32* Xz,
                              const BroadcastInfo& info, const size_t total) {
    kernels::general_broadcast<<<grid1d(total), BLOCK_SIZE_1D>>>(
        Xx, Xy, Xz, info, total, ops::Multiplication{});
}

void Broadcast::general_div(const f32* Xx, const f32* Xy, f32* Xz,
                              const BroadcastInfo& info, const size_t total) {
    kernels::general_broadcast<<<grid1d(total), BLOCK_SIZE_1D>>>(
        Xx, Xy, Xz, info, total, ops::Division{});
}