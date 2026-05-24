//
// Created by muham on 23.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SILU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SILU_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief SiLU (Sigmoid Linear Unit) activation layer.
     *
     * Also known as Swish (when β = 1). Applies the SiLU function element-wise:
     *
     *     SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *
     * SiLU is a smooth, non-monotonic activation function that has shown
     * strong performance in deep networks, often outperforming ReLU in modern architectures.
     */
    class SiLU : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a SiLU layer.
         */
        SiLU();
        ~SiLU() override;

        /**
         * @brief Performs forward pass with SiLU activation.
         *
         * @param input Input tensor
         * @return Output tensor with SiLU applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * SiLU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * SiLU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SILU_HPP