//
// Created by muham on 8.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::avx2 {
    /**
     * @brief AVX2 accelerated reduction (fold) operations on contiguous float arrays.
     *
     * All functions are optimized with a hybrid approach:
     * - Main loop uses 8-wide AVX2 vectorization
     * - Remainder uses scalar operations
     */
    struct reduce {
        /**
         * @brief Computes the sum of all elements in the array.
         *
         * @param x Input array pointer.
         * @param N Number of elements to process.
         * @return Sum of all elements (`∑x[i]`).
         *
         * @note Returns 0.0f if N == 0.
         */
        [[nodiscard]]
        static f32 sum(const f32* __restrict x, size_t N);
        /**
         * @brief Computes the arithmetic mean (average) of the array.
         *
         * @param x Input array pointer.
         * @param N Number of elements.
         * @return Mean value = sum(x) / N.
         *
         * @note Returns 0.0f if N == 0.
         */
        [[nodiscard]]
        static f32 mean(const f32* __restrict x, size_t N);
        /**
         * @brief Computes the population variance of the array.
         *
         * @param x Input array pointer.
         * @param N Number of elements.
         * @return Population variance = (∑(x[i] - mean)²) / N
         *
         * @note Uses population variance (divide by N, not N-1).
         * @see std()
         */
        [[nodiscard]]
        static f32 var(const f32* __restrict x, size_t N);
        /**
         * @brief Computes the standard deviation of the array.
         *
         * @param x Input array pointer.
         * @param N Number of elements.
         * @return Standard deviation = sqrt(variance)
         *
         * @note Uses population standard deviation.
         */
        [[nodiscard]]
        static f32 std(const f32* __restrict x, size_t N);
        /**
         * @brief Finds the minimum value in the array.
         *
         * @param x Input array pointer.
         * @param N Number of elements.
         * @return Minimum value in the array.
         *
         * @warning Behavior is undefined if N == 0 (assumes N ≥ 1).
         */
        [[nodiscard]]
        static f32 min(const f32* __restrict x, size_t N);
        /**
         * @brief Finds the maximum value in the array.
         *
         * @param x Input array pointer.
         * @param N Number of elements.
         * @return Maximum value in the array.
         *
         * @warning Behavior is undefined if N == 0 (assumes N ≥ 1).
         */
        [[nodiscard]]
        static f32 max(const f32* __restrict x, size_t N);
        /**
         * @brief Computes L1 norm (Manhattan norm): ∑|x[i]|
         *
         * @param x Input array pointer.
         * @param N Number of elements.
         * @return L1 norm of the array.
         */
        [[nodiscard]]
        static f32 norm1(const f32* __restrict x, size_t N);
        /**
         * @brief Computes L2 norm (Euclidean norm): sqrt(∑x[i]²)
         *
         * @param x Input array pointer.
         * @param N Number of elements.
         * @return L2 norm of the array.
         */
        [[nodiscard]]
        static f32 norm2(const f32* __restrict x, size_t N);
        /**
         * @brief Computes the dot product of two arrays.
         *
         * @param Xx First input array.
         * @param Xy Second input array.
         * @param N  Number of elements.
         * @return Dot product = ∑(Xx[i] * Xy[i])
         *
         * @note Arrays must not overlap.
         */
        [[nodiscard]]
        static f32 dot(const f32* __restrict Xx, const f32* __restrict Xy, size_t N);
        [[nodiscard]]
        static bool equal(const f32* Xx, const f32* Xy, size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP