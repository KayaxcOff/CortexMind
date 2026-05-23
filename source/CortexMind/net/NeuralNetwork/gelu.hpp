//
// Created by muham on 23.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_GELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_GELU_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief GELU (Gaussian Error Linear Unit) activation layer.
     *
     * Applies the GELU function element-wise:
     *
     *     GELU(x) = x * Φ(x)
     *
     * where Φ(x) is the cumulative distribution function of the standard normal distribution.
     *
     * A common approximation used in practice is:
     *
     *     GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     *
     * GELU is widely used in modern transformer models (BERT, GPT, etc.) as it
     * provides smoother gradients compared to ReLU.
     */
    class GeLU : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a GELU layer.
         */
        GeLU();
        ~GeLU() override;

        /**
         * @brief Performs forward pass with GELU activation.
         *
         * @param input Input tensor
         * @return Output tensor with GELU applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * GELU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * GELU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_GELU_HPP