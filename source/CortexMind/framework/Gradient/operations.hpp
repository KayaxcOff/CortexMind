//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP

#include <CortexMind/framework/Gradient/flow.hpp>

namespace cortex::_fw::meta {
    /**
     * @brief Gradient node for addition operation (`AddBackward`).
     *
     * Computes gradients for `z = x + y` during backward pass.
     */
    struct add : GradientFlow {
        /**
         * @brief Constructs an addition gradient node.
         *
         * @param _x GradientPacked for first operand
         * @param _y GradientPacked for second operand
         */
        add(const GradientPacked& _x, const GradientPacked& _y);
        ~add() override;

        /**
         * @brief Computes gradients for both inputs.
         *
         * For `z = x + y`, we have:
         * - ∂L/∂x = ∂L/∂z
         * - ∂L/∂y = ∂L/∂z
         */
        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;   ///< First input tensor
        Tensor* ty;   ///< Second input tensor
    };

    /**
     * @brief Gradient node for subtraction operation (`SubBackward`).
     *
     * Computes gradients for `z = x - y` during backward pass.
     */
    struct sub : GradientFlow {
        /**
         * @brief Constructs an subtraction gradient node.
         *
         * @param _x GradientPacked for first operand
         * @param _y GradientPacked for second operand
         */
        sub(const GradientPacked& _x, const GradientPacked& _y);
        ~sub() override;

        /**
         * @brief Computes gradients for both inputs.
         *
         * For `z = x - y`, we have:
         * - ∂L/∂x = ∂L/∂z
         * - ∂L/∂y = -∂L/∂z
         */
        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;   ///< First input tensor
        Tensor* ty;   ///< Second input tensor
    };

    /**
     * @brief Gradient node for multiplication operation (`MulBackward`).
     *
     * Computes gradients for `z = x * y` during backward pass.
     */
    struct mul : GradientFlow {
        /**
         * @brief Constructs an multiplication gradient node.
         *
         * @param _x GradientPacked for first operand
         * @param _y GradientPacked for second operand
         */
        mul(const GradientPacked& _x, const GradientPacked& _y);
        ~mul() override;

        /**
         * @brief Computes gradients for both inputs.
         *
         * For `z = x * y`, we have:
         * - ∂L/∂x = ∂L/∂z * y
         * - ∂L/∂y = ∂L/∂z * x
         */
        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;   ///< First input tensor
        Tensor* ty;   ///< Second input tensor
    };

    /**
     * @brief Gradient node for division operation (`DivBackward`).
     *
     * Computes gradients for `z = x / y` during backward pass.
     */
    struct div : GradientFlow {
        /**
         * @brief Constructs an division gradient node.
         *
         * @param _x GradientPacked for first operand
         * @param _y GradientPacked for second operand
         */
        div(const GradientPacked& _x, const GradientPacked& _y);
        ~div() override;

        /**
         * @brief Computes gradients for both inputs.
         *
         * For `z = x / y`, we have:
         * - ∂L/∂x = ∂L/∂z / y
         * - ∂L/∂y = -∂L/∂z * (x / y²)
         */
        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;   ///< First input tensor
        Tensor* ty;   ///< Second input tensor
    };

    /**
     * @brief Gradient node for sum reduction operation (`SumBackward`).
     *
     * Computes gradients for `z = sum(x)` during backward pass.
     */
    struct sum : GradientFlow {
        /**
         * @brief Constructs a sum gradient node.
         *
         * @param _x Input tensor packed data
         */
        explicit sum(const GradientPacked& _x);
        ~sum() override;

        /**
         * @brief Computes gradient for the input.
         *
         * For `z = sum(x)`, we have:
         * - ∂L/∂x = ∂L/∂z * ones_like(x)
         */
        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;   ///< Input tensor
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP