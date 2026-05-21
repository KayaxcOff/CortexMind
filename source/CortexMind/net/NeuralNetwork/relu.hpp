//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief ReLU (Rectified Linear Unit) activation layer.
     *
     * Applies the ReLU function element-wise: `f(x) = max(0, x)`
     *
     * This is one of the most commonly used activation functions in deep learning
     * due to its simplicity and effectiveness in mitigating vanishing gradients.
     */
    class ReLU : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a ReLU layer.
         */
        ReLU();
        ~ReLU() override;

        /**
         * @brief Performs forward pass with ReLU activation.
         *
         * @param input Input tensor
         * @return Output tensor with ReLU applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * ReLU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * ReLU layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP