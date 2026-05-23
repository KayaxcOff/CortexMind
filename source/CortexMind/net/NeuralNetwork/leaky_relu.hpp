//
// Created by muham on 23.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Leaky ReLU activation layer.
     *
     * Applies the Leaky ReLU function element-wise:
     *
     *     LeakyReLU(x) = x          if x > 0
     *                  = alpha * x  if x ≤ 0
     *
     * This variant of ReLU allows a small, non-zero gradient for negative inputs,
     * helping to mitigate the "dying ReLU" problem.
     */
    class LeakyReLU : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a LeakyReLU layer.
         *
         * @param alpha Negative slope coefficient (default: 0.01)
         */
        explicit LeakyReLU(float32 alpha = 0.01f);
        ~LeakyReLU() override;

        /**
         * @brief Performs forward pass with Leaky ReLU activation.
         *
         * @param input Input tensor
         * @return Output tensor with Leaky ReLU applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * LeakyReLU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * LeakyReLU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;

        /**
         * @brief Updates the negative slope coefficient (alpha).
         *
         * @param _alpha New alpha value
         */
        void Set(float32 _alpha = 0.01f);
    private:
        float32 alpha;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP