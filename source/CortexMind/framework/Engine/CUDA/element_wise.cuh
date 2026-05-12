//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH

#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::cuda {
    /**
     * @brief CUDA accelerated element-wise mathematical functions.
     *
     * All functions are unary (operate on single input array) and launch
     * the generic `kernels::activation` kernel with appropriate functors
     * from the `ops` namespace.
     */
    struct ElementWise {
        /**
         * @brief Element-wise power operation: `Z[i] = X[i] ^ exp`
         *
         * @param Xx  Input array
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void pow(const f32* __restrict Xx, f32 exp, f32* __restrict Xz, size_t N);

        /**
         * @brief Element-wise square root: `Z[i] = sqrt(X[i])`
         *
         * @param Xx  Input array
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);

        /**
         * @brief Element-wise reciprocal square root: `Z[i] = 1 / sqrt(X[i])`
         *
         * @param Xx  Input array
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void rsqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);

        /**
         * @brief Element-wise square: `Z[i] = X[i]²`
         *
         * @param Xx  Input array
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void square(const f32* __restrict Xx, f32* __restrict Xz, size_t N);

        /**
         * @brief Element-wise natural logarithm: `Z[i] = log(X[i])`
         *
         * @param Xx  Input array
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t N);

        /**
         * @brief Element-wise exponential: `Z[i] = e^X[i]`
         *
         * @param Xx  Input array
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t N);

        /**
         * @brief Element-wise absolute value: `Z[i] = |X[i]|`
         *
         * @param Xx  Input array
         * @param exp Exponent value (broadcasted)
         * @param Xz  Output array
         * @param N   Number of elements
         */
        static void abs(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH