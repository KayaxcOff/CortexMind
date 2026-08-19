//
// Created by muham on 6.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP

#include <CortexMind/framework/Tools/view.hpp>

namespace cortex::_fw::avx2 {
   /**
     * @brief Host-side dispatch interface for scalar CPU operations.
     *
     * Provides element-wise scalar arithmetic operations for tensors represented
     * by @ref TensorView. The supported data type is determined at runtime from
     * the tensor's @ref DType and dispatched to the corresponding CPU kernel.
     *
     * Scalar values are provided as float on the host side and converted to the
     * tensor's native element type before the kernel is launched.
     *
     * The class provides both out-of-place operations, which read from an input
     * tensor and write the result to a separate output tensor, and in-place
     * operations, which modify the input tensor directly.
     */
    struct ScalarOp {
        /**
         * @brief Adds a scalar value to every element of a tensor.
         *
         * The operation is performed element-wise and the result is written to
         * the output tensor.
         *
         * @param Xx Input tensor.
         * @param value Scalar value to add.
         * @param Xz Output tensor receiving the result.
         */
        static void add(const TensorView& Xx, float value, TensorView& Xz);

        /**
         * @brief Subtracts a scalar value from every element of a tensor.
         *
         * The operation is performed element-wise and the result is written to
         * the output tensor.
         *
         * @param Xx Input tensor.
         * @param value Scalar value to subtract.
         * @param Xz Output tensor receiving the result.
         */
        static void sub(const TensorView& Xx, float value, TensorView& Xz);

        /**
         * @brief Multiplies every element of a tensor by a scalar value.
         *
         * The operation is performed element-wise and the result is written to
         * the output tensor.
         *
         * @param Xx Input tensor.
         * @param value Scalar multiplier.
         * @param Xz Output tensor receiving the result.
         */
        static void mul(const TensorView& Xx, float value, TensorView& Xz);

        /**
         * @brief Divides every element of a tensor by a scalar value.
         *
         * The operation is performed element-wise and the result is written to
         * the output tensor.
         *
         * @param Xx Input tensor.
         * @param value Scalar divisor.
         * @param Xz Output tensor receiving the result.
         */
        static void div(const TensorView& Xx, float value, TensorView& Xz);

        /**
         * @brief Adds a scalar value to every element of a tensor in-place.
         *
         * @param Xx Tensor to modify.
         * @param value Scalar value to add.
         */
        static void add(TensorView& Xx, float value);

        /**
         * @brief Subtracts a scalar value from every element of a tensor in-place.
         *
         * @param Xx Tensor to modify.
         * @param value Scalar value to subtract.
         */
        static void sub(TensorView& Xx, float value);

        /**
         * @brief Multiplies every element of a tensor by a scalar value in-place.
         *
         * @param Xx Tensor to modify.
         * @param value Scalar multiplier.
         */
        static void mul(TensorView& Xx, float value);

        /**
         * @brief Divides every element of a tensor by a scalar value in-place.
         *
         * @param Xx Tensor to modify.
         * @param value Scalar divisor.
         */
        static void div(TensorView& Xx, float value);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP