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

    /**
     * @brief Gradient node for matrix multiplication operation (`MatMulBackward`).
     *
     * Computes gradients for `Z = X @ Y` during the backward pass.
     *
     * Mathematical rules used:
     * - If `Z = X @ Y`, then:
     *   - ∂L/∂X = ∂L/∂Z @ Y^T
     *   - ∂L/∂Y = X^T @ ∂L/∂Z
     */
    struct matmul : GradientFlow {
        /**
         * @brief Constructs a matrix multiplication gradient node.
         *
         * @param _x GradientPacked for left matrix (X)
         * @param _y GradientPacked for right matrix (Y)
         */
        matmul(const GradientPacked& _x, const GradientPacked& _y);
        ~matmul() override;

        /**
         * @brief Computes gradients with respect to both input matrices.
         *
         * For `Z = X @ Y`:
         * - Gradient w.r.t. X: `grad_Z @ Y^T`
         * - Gradient w.r.t. Y: `X^T @ grad_Z`
         *
         * @param _grad Gradient of the output tensor (∂L/∂Z)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;   ///< Left input matrix (X)
        Tensor* ty;   ///< Right input matrix (Y)
    };

    struct pow : GradientFlow {
        pow(const GradientPacked& _x, f32 _exp);
        ~pow() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 exponent;
    };

    struct sqrt : GradientFlow {
        explicit sqrt(const GradientPacked& _x);
        ~sqrt() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct exp : GradientFlow {
        explicit exp(const GradientPacked& _x);
        ~exp() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct log : GradientFlow {
        explicit log(const GradientPacked& _x);
        ~log() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct rsqrt : GradientFlow {
        explicit rsqrt(const GradientPacked& _x);
        ~rsqrt() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct sin : GradientFlow {
        explicit sin(const GradientPacked& _x);
        ~sin() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct cos : GradientFlow {
        explicit cos(const GradientPacked& _x);
        ~cos() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct abs : GradientFlow {
        explicit abs(const GradientPacked& _x);
        ~abs() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct neg : GradientFlow {
        explicit neg(const GradientPacked& _x);
        ~neg() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP