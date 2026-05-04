//
// Created by muham on 3.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Fully connected (Dense) layer.
     *
     * Implements a standard linear transformation: `output = input × weight + bias`
     * with support for different device types (CPU / CUDA).
     */
    class Dense : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Dense layer.
         *
         * @param in_dim Input feature dimension
         * @param out_dim Output feature dimension
         * @param d_type Device type (host or cuda)
         */
        Dense(int64 in_dim, int64 out_dim, _fw::sys::deviceType d_type = _fw::sys::host);
        ~Dense() override;

        /**
         * @brief Forward pass: computes `output = input @ weight + bias`
         *
         * @param input Input tensor of shape `(batch_size, in_dim)` or `(in_dim)`
         * @return Output tensor of shape `(batch_size, out_dim)` or `(out_dim)`
         */
        [[nodiscard]]
        tensor forward(tensor &input) override;
        /**
         * @brief Returns references to the trainable parameters (weight and bias).
         * @return Vector containing weight and bias tensors.
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getWeight() override;
        /**
         * @brief Returns references to the gradients of trainable parameters.
         * @return Vector containing weight.grad() and bias.grad()
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradient() override;
    private:
        tensor weight;
        tensor bias;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP