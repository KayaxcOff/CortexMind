//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_BROADCAST_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_BROADCAST_CUH

#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA broadcast operations for 2D float arrays.
     *
     * Row broadcast: Y(N) applied to each row of X(M,N)
     * Col broadcast: Y(M) applied to each column of X(M,N)
     *
     * @note Col broadcast requires M < 65535 (CUDA gridDim.y limit).
     */
    struct Broadcast {
        static void row_add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        static void row_sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        static void row_mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        static void row_div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);

        static void row_add(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);
        static void row_sub(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);
        static void row_mul(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);
        static void row_div(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        static void col_add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        static void col_sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        static void col_mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);
        static void col_div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, size_t M, size_t N);

        static void col_add(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);
        static void col_sub(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);
        static void col_mul(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);
        static void col_div(f32* Xx, const f32* __restrict Xy, size_t M, size_t N);

        static void general_add(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const BroadcastInfo& info, size_t total);
        static void general_sub(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const BroadcastInfo& info, size_t total);
        static void general_mul(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const BroadcastInfo& info, size_t total);
        static void general_div(const f32* __restrict Xx, const f32* __restrict Xy, f32* __restrict Xz, const BroadcastInfo& info, size_t total);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_BROADCAST_CUH