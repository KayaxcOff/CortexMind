//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Fully connected (dense) layer.
     *
     * This is the most common neural network layer, performing a linear transformation
     * followed by an optional bias addition.
     *
     * Mathematically: `Y = X · W^T + b`
     */
    class Dense : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Dense layer.
         *
         * @param in_dim   Number of input features
         * @param out_dim  Number of output features
         * @param device   Target device (HOST or CUDA)
         */
        Dense(int64 in_dim, int64 out_dim, _fw::sys::DeviceType device = _fw::sys::DeviceType::kHOST);
        ~Dense() override;

        /**
         * @brief Performs the forward pass of the dense layer.
         *
         * @param input Input tensor of shape `(batch_size, in_dim)`
         * @return Output tensor of shape `(batch_size, out_dim)`
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of the layer.
         *
         * @return Vector containing weight and bias tensors
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * @return Vector containing weight.grad() and bias.grad()
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;

    private:
        tensor bias, weight;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP