//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief   Flatten layer – collapses all non-batch dimensions into features
     *
     * Transforms input tensor from [batch, ...] → [batch, features]
     * where features = product of all dimensions after batch.
     */
    class Flatten : public _fw::Layer {
    public:
        /**
         * @brief   Constructs a parameter-free Flatten layer
         *
         * @note    Name is fixed to "Flatten"
         * @note    Starts in training mode (irrelevant for this layer)
         */
        Flatten();
        ~Flatten() override;

        /**
         * @brief   Forward pass: flattens input tensor to 2D
         * @param   input   Input tensor of any shape [batch, d1, d2, ..., dn]
         * @return  Flattened tensor [batch, d1×d2×...×dn]
         *
         * @note    Calls MindTensor::flat() — may copy if input non-contiguous
         * @note    Output shares storage with input when possible (view)
         */
        [[nodiscard]]
        tensor forward(tensor &input) override;
        /**
         * @brief   Returns empty list (Flatten has no learnable parameters)
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

#endif //CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP