//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief   ReLU activation layer: ReLU(x) = max(x, 0)
     *
     * Applies element-wise ReLU non-linearity. Commonly used after linear layers.
     */
    class ReLU : public _fw::Layer {
    public:
        /**
         * @brief   Constructs a parameter-free ReLU layer
         *
         * @note    Name is fixed to "ReLU"
         * @note    Starts in training mode (irrelevant for this layer)
         */
        ReLU();
        ~ReLU() override;

        /**
         * @brief   Forward pass: applies ReLU element-wise (in-place when possible)
         * @param   input   Input tensor (any shape)
         * @return  Same tensor with ReLU applied (input is modified)
         *
         * @note    Dispatches to device-specific implementation:
         *          - host → AVX2 vectorized (8 floats per iteration)
         *          - cuda → custom kernel via activation_t::relu
         * @note    Operation is in-place: input tensor is modified directly
         * @note    Registers autograd flow node if requires_grad = true
         */
        [[nodiscard]]
        tensor forward(tensor &input) override;
        /**
         * @brief   Returns empty list (ReLU has no learnable parameters)
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> parameters() override;
        /**
         * @brief   Returns empty list (no gradients to accumulate)
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> gradients() override;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP