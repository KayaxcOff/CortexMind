//
// Created by muham on 6.12.2025.
//

#ifndef CORTEXMIND_DENSE_HPP
#define CORTEXMIND_DENSE_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
        /**
        * @brief Fully connected (dense) layer for neural networks.
        *
        * This layer applies a linear transformation to the incoming data:
        * Y = X * W + b, optionally followed by an activation function.
        */
    class Dense : public _fw::Layer {
    public:
        /**
        * @brief Constructs a Dense layer.
        * @param input_size Number of input features.
        * @param output_size Number of output features.
        * @param activation Optional activation function (default is nullptr).
        */
        Dense(size_t input_size, size_t output_size, std::unique_ptr<_fw::ActivationFunc> activation = nullptr);
        ~Dense() override = default; ///< Destructor

        /**
         * @brief Forward pass through the Dense layer.
         * @param input Input tensor of shape (batch, channels, height, width) or flattenable.
         * @return Output tensor after linear transformation and optional activation.
         */
        tensor forward(const tensor &input) override;

        /**
         * @brief Backward pass computing gradients.
         * @param grad_output Gradient of the loss w.r.t. the layer's output.
         * @return Gradient of the loss w.r.t. the layer's input.
         */
        tensor backward(const tensor &grad_output) override;

        /**
         * @brief Returns a string representing the layer configuration.
         * @return String "Dense"
         */
        [[nodiscard]] std::string config() const override;

        /**
         * @brief Returns references to the gradient tensors.
         * @return Vector of references to grad_weights and grad_biases.
         */
        std::vector<std::reference_wrapper<tensor>> gradients() override;

        /**
         * @brief Returns references to the layer's parameters.
         * @return Vector of references to weights and biases.
         */
        std::vector<std::reference_wrapper<tensor>> parameters() override;
    private:
        size_t in_size, out_size; ///< Input and output feature sizes
    };
}

#endif //CORTEXMIND_DENSE_HPP