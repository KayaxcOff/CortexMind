//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Sigmoid activation layer.
     *
     * Applies the sigmoid function element-wise:
     *
     *     sigmoid(x) = 1 / (1 + exp(-x))
     *
     * Output range is `(0, 1)`. Commonly used in the final layer of binary
     * classification models to produce probability-like outputs.
     */
    class Sigmoid : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Sigmoid layer.
         */
        Sigmoid();
        ~Sigmoid() override;

        /**
         * @brief Performs forward pass with Sigmoid activation.
         *
         * @param input Input tensor
         * @return Output tensor with sigmoid applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * Sigmoid layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * Sigmoid layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_HPP