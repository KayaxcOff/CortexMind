//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_ELEM_WISE_H
#define CORTEXMIND_CORE_ENGINE_CUDA_ELEM_WISE_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief High-level interface for CUDA element-wise mathematical operations.
     *
     * Provides optimized GPU implementations of common unary and scalar mathematical
     * functions using vectorized kernels (float4).
     */
    struct ElementWise {
        /**
         * @brief Element-wise power: Z[i] = Xx[i] ^ exp
         * @param Xx  Input array
         * @param exp Scalar exponent
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void pow(const f32* __restrict Xx, f32 exp, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise square root: Z[i] = sqrt(Xx[i])
         * @param Xx  Input array
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise square: Z[i] = Xx[i]²
         * @param Xx  Input array
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void square(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise natural logarithm: Z[i] = log(Xx[i])
         * @param Xx  Input array
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        /**
         * @brief Element-wise exponential: Z[i] = exp(Xx[i])
         * @param Xx  Input array
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_ELEM_WISE_H