//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_REDUCE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_REDUCE_HPP

#include <CortexMind/framework/Storage/stor.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/reduce.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <vector>

namespace cortex::_fw::ix {
    /**
     * @brief Reduction operations dispatcher.
     *
     * Provides high-level access to common reduction operations such as sum,
     * mean, variance, norms, min/max and dot product. Automatically dispatches
     * to the appropriate backend based on the tensor's device.
     */
    class reduce {
    public:
        reduce();
        ~reduce();

        /**
         * @brief Computes the sum of all elements in the tensor.
         */
        [[nodiscard]]
        static f32 sum(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Computes the arithmetic mean of the tensor.
         */
        [[nodiscard]]
        static f32 mean(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Computes the population variance of the tensor.
         */
        [[nodiscard]]
        static f32 var(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Computes the standard deviation of the tensor.
         */
        [[nodiscard]]
        static f32 stdv(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Finds the minimum value in the tensor.
         */
        [[nodiscard]]
        static f32 min(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Finds the maximum value in the tensor.
         */
        [[nodiscard]]
        static f32 max(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Computes L1 norm (∑|x[i]|).
         */
        [[nodiscard]]
        static f32 norm1(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Computes L2 norm (Euclidean norm).
         */
        [[nodiscard]]
        static f32 norm2(const TensorStorage* __restrict x, size_t N);
        /**
         * @brief Computes dot product between two tensors: `sum(Xx[i] * Xy[i])`
         */
        [[nodiscard]]
        static f32 dot(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, size_t N);

        static void sum(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, const std::vector<i64>& shape, const std::vector<i64>& dims, const std::vector<i64>& out_shape);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_REDUCE_HPP