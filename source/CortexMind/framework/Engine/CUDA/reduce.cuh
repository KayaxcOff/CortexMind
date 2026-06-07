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
        static void sum(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void mean(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void var(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void stdv(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void argmax(const f32* __restrict Xx, i32* __restrict Xz, size_t N);
        static void argmin(const f32* __restrict Xx, i32* __restrict Xz, size_t N);

        static void sum_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner);
        static void mean_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner);
        static void var_dim(const f32* __restrict Xx, f32* __restrict Xz, const f32* __restrict means, size_t outer, size_t dim, size_t inner);
        static void stdv_dim(const f32* __restrict Xx, f32* __restrict Xz, const f32* __restrict means, size_t outer, size_t dim, size_t inner);
        static void min_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner);
        static void max_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner);
        static void argmax_dim(const f32* __restrict Xx, i32* __restrict Xz, size_t outer, size_t dim, size_t inner);
        static void argmin_dim(const f32* __restrict Xx, i32* __restrict Xz, size_t outer, size_t dim, size_t inner);
        static void norm1_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner);
        static void norm2_dim(const f32* __restrict Xx, f32* __restrict Xz, size_t outer, size_t dim, size_t inner);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_REDUCE_CUH