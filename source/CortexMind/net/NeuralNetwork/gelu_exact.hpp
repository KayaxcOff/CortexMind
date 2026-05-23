//
// Created by muham on 23.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_GELU_EXACT_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_GELU_EXACT_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Exact GELU (Gaussian Error Linear Unit) activation layer.
     *
     * Applies the mathematically exact GELU function element-wise:
     *
     *     GELU(x) = 0.5 * x * (1 + erf(x / √2))
     *
     * where `erf` is the error function.
     *
     * This is more accurate than the approximation used in `GeLU`, but slightly
     * more computationally expensive. Commonly used in research and high-precision models.
     */
    class GeLUExact : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a GeLUExact layer.
         */
        GeLUExact();
        ~GeLUExact() override;

        /**
         * @brief Performs forward pass with exact GELU activation.
         *
         * @param input Input tensor
         * @return Output tensor with exact GELU applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * GeLUExact layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * GeLUExact layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_GELU_EXACT_HPP