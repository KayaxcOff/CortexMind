//
// Created by muham on 14.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_ACTIVATION_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_ACTIVATION_OPERATIONS_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::disp {
    /**
     * @brief Dispatch manager for activation functions.
     *
     * This class routes activation operations to the appropriate backend
     * implementation depending on the currently active device (Host or CUDA).
     */
    class Activation {
    public:
        /**
         * @brief Constructs an Activation dispatcher for the specified device.
         *
         * @param _d_type Target device type (HOST or CUDA)
         */
        explicit Activation(sys::DeviceType _d_type);
        ~Activation();

        /**
         * @brief Changes the active device for subsequent activation operations.
         *
         * @param _d_type New target device
         */
        void SetDevice(sys::DeviceType _d_type);

        /**
         * @brief ReLU activation: `Z[i] = max(0, X[i])`
         */
        void relu(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Leaky ReLU activation.
         *
         * @param alpha Negative slope coefficient (default = 0.01)
         */
        void leaky_relu(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N, f32 alpha = 0.01f) const;
        /**
         * @brief Sigmoid activation: `Z[i] = 1 / (1 + exp(-X[i]))`
         */
        void sigmoid(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Fast sigmoid approximation.
         */
        void sigmoid_fast(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Hyperbolic tangent activation: `Z[i] = tanh(X[i])`
         */
        void tanh(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief GELU activation (tanh approximation).
         */
        void gelu(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief GELU activation using erf function (more accurate).
         */
        void gelu_exact(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief SiLU (Sigmoid Linear Unit) activation.
         */
        void silu(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Fast SiLU approximation.
         */
        void silu_fast(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        /**
         * @brief Swish activation: `Z[i] = X[i] * sigmoid(beta * X[i])`
         *
         * @param beta Scaling parameter (default = 1.0)
         */
        void swish(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N, f32 beta = 1.0f) const;
        /**
         * @brief Fast Swish approximation.
         *
         * @param beta Scaling parameter (default = 1.0)
         */
        void swish_fast(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N, f32 beta = 1.0f) const;
    private:
        sys::DeviceType d_type;
    };
} // namespace cortex::_fw::disp

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_ACTIVATION_OPERATIONS_HPP