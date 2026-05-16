//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_ELEMENT_WISE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_ELEMENT_WISE_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    /**
     * @brief Element-wise unary operations dispatcher.
     *
     * Provides high-level access to common mathematical functions such as
     * power, square root, logarithm, exponential, absolute value, and trigonometric
     * functions with automatic device dispatching.
     */
    class ElementWise {
    public:
        ElementWise();
        ~ElementWise();

        /**
         * @brief Element-wise power: `Z[i] = X[i] ^ exp`
         */
        static void pow(const TensorStorage* __restrict Xx, f32 exp, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise square root: `Z[i] = sqrt(X[i])`
         */
        static void sqrt(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise natural logarithm: `Z[i] = log(X[i])`
         */
        static void log(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise exponential: `Z[i] = exp(X[i])`
         */
        static void exp(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise absolute value: `Z[i] = |X[i]|`
         */
        static void abs(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise sine: `Z[i] = sin(X[i])`
         */
        static void sin(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise cosine: `Z[i] = cos(X[i])`
         */
        static void cos(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise reciprocal square root: `Z[i] = 1 / sqrt(X[i])`
         */
        static void rsqrt(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise negation: `Z[i] = -X[i]`
         */
        static void neg(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Element-wise sign function: `Z[i] = sign(X[i])` (-1, 0, or 1)
         */
        static void sign(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_ELEMENT_WISE_HPP