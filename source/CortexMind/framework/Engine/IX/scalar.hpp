//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_SCALAR_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    /**
     * @brief Scalar operations dispatcher (element-wise + scalar value).
     *
     * Provides both out-of-place and in-place scalar arithmetic operations
     * (`add`, `sub`, `mul`, `div`) with automatic device dispatching.
     */
    class ScalarOp {
    public:
        ScalarOp();
        ~ScalarOp();

        /**
         * @brief Out-of-place addition: `Z = X + value`
         */
        static void add(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place subtraction: `Z = X - value`
         */
        static void sub(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place multiplication: `Z = X * value`
         */
        static void mul(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N);
        /**
         * @brief Out-of-place division: `Z = X / value`
         */
        static void div(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N);

        /**
         * @brief In-place addition: `X = X + value`
         */
        static void add(TensorStorage* Xx, f32 value, size_t N);
        /**
         * @brief In-place subtraction: `X = X - value`
         */
        static void sub(TensorStorage* Xx, f32 value, size_t N);
        /**
         * @brief In-place multiplication: `X = X * value`
         */
        static void mul(TensorStorage* Xx, f32 value, size_t N);
        /**
         * @brief In-place division: `X = X / value`
         */
        static void div(TensorStorage* Xx, f32 value, size_t N);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_SCALAR_HPP