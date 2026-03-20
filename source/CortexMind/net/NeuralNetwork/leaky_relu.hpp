//
// Created by muham on 19.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief   Leaky ReLU activation layer
     *
     * Applies element-wise Leaky ReLU with fixed negative slope α.
     * Helps mitigate dying ReLU problem by allowing small gradient for negative inputs.
     */
    class LeakyReLU : public _fw::Layer {
    public:
        /**
         * @brief   Constructs LeakyReLU layer with given leakage coefficient
         * @param   alpha   Negative slope for x ≤ 0 (typically 0.01 or 0.1)
         *
         * @note    Name is fixed to "LeakyReLU"
         * @note    Starts in training mode (irrelevant for this layer)
         * @note    Alpha is stored as member (fixed, not learnable)
         */
        explicit LeakyReLU(float32 alpha);
        ~LeakyReLU() override;

        /**
         * @brief   Forward pass: applies Leaky ReLU element-wise (in-place when possible)
         * @param   input   Input tensor (any shape)
         * @return  Same tensor with Leaky ReLU applied (input is modified)
         *
         * @note    Dispatches to device-specific implementation:
         *          - host → AVX2 vectorized leaky_relu
         *          - cuda → custom kernel via activation_t::leaky_relu
         * @note    Operation is in-place: input tensor is modified directly
         * @note    Registers autograd flow node if requires_grad = true
         */
        [[nodiscard]]
        tensor forward(tensor &input) override;
        /**
         * @brief   Returns empty list (LeakyReLU has no learnable parameters)
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> parameters() override;
        /**
         * @brief   Returns empty list (no gradients to accumulate)
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        float32 alpha;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP