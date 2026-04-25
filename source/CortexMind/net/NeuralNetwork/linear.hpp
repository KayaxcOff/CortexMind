//
// Created by muham on 25.04.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_LINEAR_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_LINEAR_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Memory/device.hpp>

namespace cortex::nn {
    /**
     * @brief Fully connected (dense) linear layer.
     *
     * Implements the transformation: `output = input @ weight + bias`
     * where `@` denotes matrix multiplication.
     *
     * Supports both CPU and CUDA backends and automatic gradient tracking.
     */
    class Linear : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Linear layer.
         * @param input_dim   Size of each input sample
         * @param output_dim  Size of each output sample
         * @param d_type      Device type (host or cuda)
         */
        Linear(int64 input_dim, int64 output_dim, _fw::sys::deviceType d_type = _fw::sys::host);
        ~Linear() override;

        /**
         * @brief Performs the forward pass: `output = input @ weight + bias`
         * @param input Input tensor (usually of shape [batch_size, input_dim])
         * @return Output tensor of shape [batch_size, output_dim]
         */
        [[nodiscard]]
        tensor forward(tensor &input) override;
        /**
         * @brief Returns references to the trainable parameters (weight and bias).
         * @return Vector containing references to weight and bias tensors
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getWeight() override;
        /**
         * @brief Returns references to the gradients of the trainable parameters.
         * @return Vector containing references to weight.grad() and bias.grad()
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradient() override;
    private:
        tensor weight;
        tensor bias;

        int64 input_dim;
        int64 output_dim;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_LINEAR_HPP