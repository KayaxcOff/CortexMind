//
// Created by muham on 23.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SILU_FAST_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SILU_FAST_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Fast SiLU (Sigmoid Linear Unit) activation layer.
     *
     * Applies a fast approximation of the SiLU function element-wise:
     *
     *     SiLUFast(x) = x * sigmoid_fast(x) = x / (1 + exp(-x))
     *
     * This version uses faster intrinsics (e.g. `__expf`) for improved performance
     * at the cost of slight numerical precision compared to the standard `SiLU`.
     */
    class SiLUFast : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a SiLUFast layer.
         */
        SiLUFast();
        ~SiLUFast() override;

        /**
         * @brief Performs forward pass with fast SiLU activation.
         *
         * @param input Input tensor
         * @return Output tensor with fast SiLU applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * SiLUFast layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * SiLUFast layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SILU_FAST_HPP