//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_REDUCE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_REDUCE_CUH

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA reduction operations manager for Tensor-to-Tensor operations.
     * All outputs are written directly to device memory pointers (Xz).
     */
    struct ReduceOp {
        static void sum(const f32* __restrict Xx, f32* __restrict Xz, size_t N) const;
        static void mean(const f32* __restrict Xx, f32* __restrict Xz, size_t N) const;
        static void var(const f32* __restrict Xx, f32* __restrict Xz, size_t N) const;
        static void stdv(const f32* __restrict Xx, f32* __restrict Xz, size_t N) const;
        static void argmax(const f32* __restrict Xx, i32* __restrict Xz, size_t N) const;
        static void argmin(const f32* __restrict Xx, i32* __restrict Xz, size_t N) const;

        static void sum_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
        static void mean_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
        static void var_dim(const f32* __restrict Xx, f32* __restrict Xz, const f32* __restrict means, size_t outer, size_t dim, size_t inner) const;
        static void stdv_dim(const f32* __restrict Xx, f32* __restrict Xz, const f32* __restrict means, size_t outer, size_t dim, size_t inner) const;
        static void min_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
        static void max_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
        static void argmax_dim(const f32* __restrict Xx, i32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
        static void argmin_dim(const f32* __restrict Xx, i32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
        static void norm1_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
        static void norm2_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner) const;
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_REDUCE_CUH