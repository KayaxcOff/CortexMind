//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_REDUCE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_REDUCE_CUH

#include <CortexMind/framework/Tools/err.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA reduction operations manager.
     *
     * This class encapsulates reduction operations on GPU memory using:
     * - Custom warp-shuffle + shared memory kernels for sum and variance
     * - cuBLAS for high-performance norms, dot product, and argmin/argmax
     *
     * Uses pinned + mapped memory for efficient host-device data transfer.
     */
    struct ReduceOp {
        ReduceOp();
        ~ReduceOp();

        /**
         * @brief Computes the sum of all elements.
         * @param x Input array on device
         * @param N Number of elements
         * @return Sum of all elements
         */
        [[nodiscard]]
        f32 sum(const f32* __restrict x, size_t N);

        /**
         * @brief Computes the arithmetic mean.
         * @param x Input array on device
         * @param N Number of elements
         * @return Mean value = sum(x) / N
         */
        [[nodiscard]]
        f32 mean(const f32* __restrict x, size_t N);

        /**
         * @brief Computes the population variance.
         * @param x Input array on device
         * @param N Number of elements
         * @return Population variance = (∑(x[i] - mean)²) / N
         */
        [[nodiscard]]
        f32 var(const f32* __restrict x, size_t N);

        /**
         * @brief Computes the standard deviation.
         * @param x Input array on device
         * @param N Number of elements
         * @return Standard deviation = sqrt(variance)
         */
        [[nodiscard]]
        f32 stdv(const f32* __restrict x, size_t N);

        /**
         * @brief Finds the minimum value using cuBLAS.
         * @param x Input array on device
         * @param N Number of elements
         * @return Minimum value in the array
         */
        [[nodiscard]]
        f32 min(const f32* __restrict x, size_t N) const;

        /**
         * @brief Finds the maximum value using cuBLAS.
         * @param x Input array on device
         * @param N Number of elements
         * @return Maximum value in the array
         */
        [[nodiscard]]
        f32 max(const f32* __restrict x, size_t N) const;

        /**
         * @brief Computes L1 norm (∑|x[i]|) using cuBLAS.
         */
        [[nodiscard]]
        f32 norm1(const f32* __restrict x, size_t N) const;

        /**
         * @brief Computes L2 norm (Euclidean norm) using cuBLAS.
         */
        [[nodiscard]]
        f32 norm2(const f32* __restrict x, size_t N) const;

        /**
         * @brief Computes dot product using cuBLAS: x · y
         */
        [[nodiscard]]
        f32 dot(const f32* __restrict Xx, const f32* __restrict Xy, size_t N) const;
    private:
        f32* host_output{nullptr};   ///< Pinned + mapped host memory
        f32* cuda_output{nullptr};   ///< Device pointer (mapped)

        i32* host_index{nullptr};    ///< Pinned + mapped index memory
        i32* cuda_index{nullptr};    ///< Device pointer for argmin/argmax
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_REDUCE_CUH