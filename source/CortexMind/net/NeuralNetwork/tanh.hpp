//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_TANH_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_TANH_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Tanh (Hyperbolic Tangent) activation layer.
     *
     * Applies the hyperbolic tangent function element-wise:
     *
     *     tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     *
     * Output range is `[-1, 1]`. Often used in recurrent networks and hidden layers
     * where zero-centered output is preferred over ReLU.
     */
    class Tanh : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Tanh layer.
         */
        Tanh();
        ~Tanh() override;

        /**
         * @brief Performs forward pass with Tanh activation.
         *
         * @param input Input tensor
         * @return Output tensor with tanh applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * Tanh layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * Tanh layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_TANH_HPP