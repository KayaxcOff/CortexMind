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

    /**
     * @brief Gradient node for power operation (`PowBackward`).
     *
     * Computes gradients for `z = x ^ exp` during backward pass.
     */
    struct pow : GradientFlow {
        pow(const GradientPacked& _x, f32 _exp);
        ~pow() override;

        /**
         * @brief Computes gradient for input.
         *
         * For `z = x^exp`, we have:
         * - ∂L/∂x = ∂L/∂z * exp * x^(exp-1)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 exponent;
    };

    /**
     * @brief Gradient node for square root operation (`SqrtBackward`).
     */
    struct sqrt : GradientFlow {
        explicit sqrt(const GradientPacked& _x);
        ~sqrt() override;

        /**
         * @brief Computes gradient for input.
         *
         * For `z = sqrt(x)`, we have:
         * - ∂L/∂x = ∂L/∂z / (2 * sqrt(x))
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for exponential operation (`ExpBackward`).
     */
    struct exp : GradientFlow {
        explicit exp(const GradientPacked& _x);
        ~exp() override;

        /**
         * @brief Computes gradient for input.
         *
         * For `z = exp(x)`, we have:
         * - ∂L/∂x = ∂L/∂z * exp(x)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for natural logarithm operation (`LogBackward`).
     */
    struct log : GradientFlow {
        explicit log(const GradientPacked& _x);
        ~log() override;

        /**
         * @brief Computes gradient for input.
         *
         * For `z = log(x)`, we have:
         * - ∂L/∂x = ∂L/∂z / x
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for reciprocal square root operation (`RsqrtBackward`).
     */
    struct rsqrt : GradientFlow {
        explicit rsqrt(const GradientPacked& _x);
        ~rsqrt() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for sine operation (`SinBackward`).
     */
    struct sin : GradientFlow {
        explicit sin(const GradientPacked& _x);
        ~sin() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for cosine operation (`CosBackward`).
     */
    struct cos : GradientFlow {
        explicit cos(const GradientPacked& _x);
        ~cos() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for absolute value operation (`AbsBackward`).
     */
    struct abs : GradientFlow {
        explicit abs(const GradientPacked& _x);
        ~abs() override;

        /**
         * @brief Computes gradient for input.
         *
         * For `z = |x|`, we have:
         * - ∂L/∂x = ∂L/∂z * sign(x)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for negation operation (`NegBackward`).
     */
    struct neg : GradientFlow {
        explicit neg(const GradientPacked& _x);
        ~neg() override;

        /**
         * @brief Computes gradient for input.
         *
         * For `z = -x`, we have:
         * - ∂L/∂x = -∂L/∂z
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for scalar addition (`AddScalarBackward`).
     */
    struct add_scalar : GradientFlow {
        add_scalar(const GradientPacked& _x, f32 _scalar);
        ~add_scalar() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 scalar;
    };

    /**
     * @brief Gradient node for scalar subtraction (`SubScalarBackward`).
     */
    struct sub_scalar : GradientFlow {
        sub_scalar(const GradientPacked& _x, f32 _scalar);
        ~sub_scalar() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 scalar;
    };

    /**
     * @brief Gradient node for scalar multiplication (`MulScalarBackward`).
     */
    struct mul_scalar : GradientFlow {
        mul_scalar(const GradientPacked& _x, f32 _scalar);
        ~mul_scalar() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 scalar;
    };

    /**
     * @brief Gradient node for scalar division (`DivScalarBackward`).
     */
    struct div_scalar : GradientFlow {
        div_scalar(const GradientPacked& _x, f32 _scalar);
        ~div_scalar() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 scalar;
    };

    /**
     * @brief Gradient node for ReLU activation function (`ReLUBackward`).
     *
     * Computes gradients for `Z = ReLU(X)` during the backward pass.
     *
     * Mathematical rule:
     * - If `Z = max(0, X)`, then:
     *   - ∂L/∂X = ∂L/∂Z  if X > 0
     *   - ∂L/∂X = 0      if X ≤ 0
     */
    struct relu : GradientFlow {
        /**
         * @brief Constructs a ReLU gradient node.
         *
         * @param _x Input tensor packed data (the tensor to which ReLU was applied)
         */
        explicit relu(const GradientPacked& _x);
        ~relu() override;

        /**
         * @brief Computes gradient with respect to the input of ReLU.
         *
         * Applies the ReLU derivative (which is a binary mask: 1 if input > 0, else 0)
         * and multiplies it with the incoming gradient.
         *
         * @param _grad Gradient of the output tensor (∂L/∂Z)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    /**
     * @brief Gradient node for Tanh activation function (`TanhBackward`).
     *
     * Computes gradients for `Z = tanh(X)` during the backward pass.
     *
     * Mathematical rule:
     * - ∂L/∂X = ∂L/∂Z * (1 - tanh²(X)) = ∂L/∂Z * (1 - Z²)
     */
    struct tanh : GradientFlow {
        /**
         * @brief Constructs a Tanh gradient node.
         *
         * @param _x     Input tensor packed data (before tanh)
         * @param _y     Output tensor packed data (after tanh)
         */
        explicit tanh(const GradientPacked& _x, const GradientPacked& _y);
        ~tanh() override;

        /**
         * @brief Computes gradient with respect to the input of tanh.
         *
         * Uses the derivative: `1 - output²`
         *
         * @param _grad Gradient of the output tensor (∂L/∂Z)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    /**
     * @brief Gradient node for Sigmoid activation function (`SigmoidBackward`).
     *
     * Computes gradients for `Z = sigmoid(X)` during the backward pass.
     *
     * Mathematical rule:
     * - ∂L/∂X = ∂L/∂Z * Z * (1 - Z)
     */
    struct sigmoid : GradientFlow {
        /**
         * @brief Constructs a Sigmoid gradient node.
         *
         * @param _x     Input tensor packed data (before sigmoid)
         * @param _y     Output tensor packed data (after sigmoid)
         */
        explicit sigmoid(const GradientPacked& _x, const GradientPacked& _y);
        ~sigmoid() override;

        /**
         * @brief Computes gradient with respect to the input of sigmoid.
         *
         * Uses the derivative: `output * (1 - output)`
         *
         * @param _grad Gradient of the output tensor (∂L/∂Z)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    /**
     * @brief Gradient node for GELU activation function (`GELUBackward`).
     *
     * Computes gradients for `Z = GELU(X)` during the backward pass.
     *
     * Mathematical derivative used:
     *
     *     GELU'(x) = Φ(x) + x * φ(x)
     *
     * where:
     * - Φ(x) is the cumulative distribution function (CDF) of the standard normal
     * - φ(x) is the probability density function (PDF) of the standard normal
     *
     * This implementation uses the common approximation for numerical stability.
     */
    struct gelu : GradientFlow {
        /**
         * @brief Constructs a GELU gradient node.
         *
         * @param _x Input tensor packed data (before GELU)
         * @param _y Output tensor packed data (after GELU) - cached for derivative
         */
        explicit gelu(const GradientPacked& _x, const GradientPacked& _y);
        ~gelu() override;

        /**
         * @brief Computes gradient with respect to the input of GELU.
         *
         * Uses the analytical derivative:
         *
         *     dGELU/dx = CDF(x) + x * PDF(x) * √(2/π) * (1 + 3 * 0.044715 * x²)
         *
         * @param _grad Gradient of the output tensor (∂L/∂Z)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    /**
     * @brief Gradient node for Leaky ReLU activation function (`LeakyReLUBackward`).
     *
     * Computes gradients for `Z = LeakyReLU(X)` during the backward pass.
     *
     * Mathematical rule:
     *
     *     LeakyReLU'(x) = 1          if x > 0
     *                   = alpha      if x ≤ 0
     */
    struct leaky_relu : GradientFlow {
        /**
         * @brief Constructs a LeakyReLU gradient node.
         *
         * @param _x    Input tensor packed data (before LeakyReLU)
         * @param alpha Negative slope coefficient used in the forward pass
         */
        explicit leaky_relu(const GradientPacked& _x, f32 alpha);
        ~leaky_relu() override;

        /**
         * @brief Computes gradient with respect to the input of LeakyReLU.
         *
         * Applies the piecewise derivative:
         * - 1.0 where input > 0
         * - alpha where input ≤ 0
         *
         * @param _grad Gradient of the output tensor (∂L/∂Z)
         */
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 alpha;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP