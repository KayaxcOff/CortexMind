//
// Created by muham on 9.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP

#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <cwchar>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2-accelerated broadcasting operations.
     *
     * Provides element-wise arithmetic operations for broadcast-compatible
     * tensors. Operations are divided into row-wise, column-wise, and
     * general broadcasting implementations depending on the memory layout
     * and broadcasting pattern.
     */
    struct Broadcast {
        /**
         * @brief Broadcasts a row vector across the rows of a matrix.
         *
         * The input tensor is treated as an `M x N` matrix while the
         * broadcast tensor contains `N` elements. Each row of the matrix
         * is combined element-wise with the same row vector.
         */
        struct row {
            static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);

            static void add(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void sub(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void mul(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void div(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
        };

        /**
         * @brief Broadcasts a column vector across the columns of a matrix.
         *
         * The input tensor is treated as an `M x N` matrix while the
         * broadcast tensor contains `M` elements. Each element of the
         * column vector is applied to the corresponding row.
         */
        struct col {
            static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);

            static void add(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void sub(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void mul(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void div(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
        };

        /**
         * @brief Performs general broadcast operations using tensor strides.
         *
         * Supports broadcast-compatible tensors with arbitrary dimensionality
         * and memory strides described by @ref BroadcastInfo.
         *
         * The implementation selects an AVX2 vectorized path when the
         * innermost dimension is contiguous and falls back to scalar
         * element-wise processing for non-contiguous layouts.
         */
        struct general {
            static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);
            static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);
            static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);
            static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);

            static void add(float* Xx, const float* __restrict Xy, const BroadcastInfo& info);
            static void sub(float* Xx, const float* __restrict Xy, const BroadcastInfo& info);
            static void mul(float* Xx, const float* __restrict Xy, const BroadcastInfo& info);
            static void div(float* Xx, const float* __restrict Xy, const BroadcastInfo& info);
        };
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP