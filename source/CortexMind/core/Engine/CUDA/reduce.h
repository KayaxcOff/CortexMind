//
// Created by muham on 9.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H
#define CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA-based reduction operations using cuBLAS.
     *
     * Provides high-performance reduction operations (sum, mean, variance, min/max, norms, dot product)
     * by leveraging cuBLAS functions and pinned host memory for efficient data transfer.
     */
    struct ReduceOp {
        ReduceOp();
        ~ReduceOp();

        /**
         * @brief Returns the sum of all elements in the array.
         */
        [[nodiscard]]
        f32 sum(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the mean (average) of all elements in the array.
         */
        [[nodiscard]]
        f32 mean(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the population variance of the array.
         */
        [[nodiscard]]
        f32 var(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the standard deviation of the array.
         */
        [[nodiscard]]
        f32 std(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the minimum value in the array.
         */
        [[nodiscard]]
        f32 min(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the maximum value in the array.
         */
        [[nodiscard]]
        f32 max(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the L1 norm (sum of absolute values).
         */
        [[nodiscard]]
        f32 norm1(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the L2 norm (Euclidean norm / magnitude).
         */
        [[nodiscard]]
        f32 norm2(const f32* __restrict x, size_t N);
        /**
         * @brief Returns the dot product of two vectors (Xx · Xy).
         */
        [[nodiscard]]
        f32 dot(const f32* __restrict Xx, const f32* __restrict Xy, size_t N);
    private:
        f32* host_output;   ///< Pinned host memory for reduction result
        f32* cuda_output;   ///< Device pointer to pinned host_output
        i32* host_index;    ///< Pinned host memory for index-based reductions (min/max)
        i32* cuda_index;    ///< Device pointer to pinned host_index
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_REDUCE_H