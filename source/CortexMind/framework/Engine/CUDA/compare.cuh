//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_COMPARE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_COMPARE_CUH

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA backend for element-wise comparison operations.
     *
     * Provides high-performance GPU implementations for comparison functions
     * using generic `kernels::matrix` with appropriate comparison functors.
     */
    struct CompareTo {
        /**
         * @brief Element-wise greater than comparison (`Z = X > Y`).
         *
         * Writes 1.0f where X[i] > Y[i], otherwise 0.0f.
         */
        static void greater(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        /**
         * @brief Element-wise less than comparison (`Z = X < Y`).
         *
         * Writes 1.0f where X[i] < Y[i], otherwise 0.0f.
         */
        static void less(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        /**
         * @brief Element-wise greater than or equal comparison (`Z = X >= Y`).
         *
         * Writes 1.0f where X[i] >= Y[i], otherwise 0.0f.
         */
        static void greater_eq(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        /**
         * @brief Element-wise less than or equal comparison (`Z = X <= Y`).
         *
         * Writes 1.0f where X[i] <= Y[i], otherwise 0.0f.
         */
        static void less_eq(const f32* Xx, const f32* Yy, f32* Xz, size_t N);
        static bool equal(const f32* Xx, const f32* Yy, size_t N); // reduce → bool
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_COMPARE_CUH