//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_FAST_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_FAST_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Fast Sigmoid activation layer.
     *
     * Applies a fast approximation of the sigmoid function element-wise:
     *
     *     sigmoid_fast(x) = 1 / (1 + exp(-x))
     *
     * This version typically uses faster intrinsics (`__expf` etc.) for better
     * performance at the cost of slight numerical precision.
     */
    class SigmoidFast : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a SigmoidFast layer.
         */
        SigmoidFast();
        ~SigmoidFast() override;

        /**
         * @brief Performs forward pass with fast sigmoid activation.
         *
         * @param input Input tensor
         * @return Output tensor with fast sigmoid applied element-wise
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * SigmoidFast layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * SigmoidFast layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_FAST_HPP