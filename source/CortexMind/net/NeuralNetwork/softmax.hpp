//
// Created by muham on 29.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SOFTMAX_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SOFTMAX_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Softmax activation layer.
     *
     * Converts a vector of raw scores (logits) into a probability distribution.
     * The output values are in the range (0, 1) and sum to 1 along the specified axis.
     *
     * Commonly used in the final layer of multi-class classification models.
     *
     * Mathematical definition:
     *
     *     softmax(x_i) = exp(x_i) / Σ exp(x_j)
     */
    class Softmax : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Softmax layer.
         */
        Softmax();
        ~Softmax() override;

        /**
         * @brief Performs forward pass with Softmax activation.
         *
         * @param input Input tensor (usually logits)
         * @return Output tensor with probabilities (sum to 1 along last dimension)
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * Softmax layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * Softmax layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SOFTMAX_HPP