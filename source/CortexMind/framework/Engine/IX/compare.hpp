//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_COMPARE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_COMPARE_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    /**
     * @brief Dispatch layer for element-wise comparison operations.
     *
     * Handles device-aware routing (AVX2 vs CUDA) for comparison functions.
     */
    struct CompareTo {
        /**
         * @brief Element-wise greater than comparison (`Z = X > Y`).
         *
         * Writes 1.0f where condition is true, 0.0f otherwise.
         *
         * @param Xx First input storage
         * @param Xy Second input storage
         * @param Xz Output storage
         * @param N  Number of elements
         */
        static void greater(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise less than comparison (`Z = X < Y`).
         *
         * Writes 1.0f where condition is true, 0.0f otherwise.
         * @param Xx First input storage
         * @param Xy Second input storage
         * @param Xz Output storage
         * @param N  Number of elements
         */
        static void less(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise greater than or equal comparison (`Z = X >= Y`).
         *
         * Writes 1.0f where condition is true, 0.0f otherwise.
         * @param Xx First input storage
         * @param Xy Second input storage
         * @param Xz Output storage
         * @param N  Number of elements
         */
        static void greater_eq(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise less than or equal comparison (`Z = X <= Y`).
         *
         * Writes 1.0f where condition is true, 0.0f otherwise.
         * @param Xx First input storage
         * @param Xy Second input storage
         * @param Xz Output storage
         * @param N  Number of elements
         */
        static void less_eq(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Checks whether two tensors are element-wise equal.
         *
         * @param Xx First input storage
         * @param Xy Second input storage
         * @param N  Number of elements
         * @return `true` if all elements are equal, `false` otherwise
         *
         * @note For CUDA tensors, data is downloaded to host before comparison.
         */
        static bool equal(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_COMPARE_HPP