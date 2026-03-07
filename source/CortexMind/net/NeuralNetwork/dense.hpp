//
// Created by muham on 25.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <CortexMind/core/Engine/Memory/device.hpp>

namespace cortex::nn {
    /**
     * @brief Fully Connected (Dense) neural network layer.
     *
     * The Dense layer implements a linear transformation of the form:
     *
     *     y = xW^T + b
     *
     * where:
     * - x is the input tensor of shape (batch_size, in_features)
     * - W is the weight matrix of shape (out_features, in_features)
     * - b is the bias vector of shape (out_features)
     * - y is the output tensor of shape (batch_size, out_features)
     *
     * This layer contains trainable parameters (weights and bias) and
     * supports gradient-based optimization when training mode is enabled.
     *
     * @note This layer performs only a linear transformation.
     *       Non-linearity must be applied separately using an activation layer.
     */
    class Dense : public _fw::Layer {
    public:
        /**
         * @brief Constructs a Dense layer.
         *
         * @param in_feats  Number of input features.
         * @param out_feats Number of output features (neurons).
         * @param _dev Target device (e.g., host or accelerator).
         */
        Dense(_fw::i64 in_feats, _fw::i64 out_feats, _fw::sys::device _dev = _fw::sys::device::host);
        /**
         * @brief Destructor.
         */
        ~Dense() override;

        /**
         * @brief Performs the forward pass.
         *
         * Computes:
         *     output = input * W^T + b
         *
         * @param input Input tensor of shape (batch_size, in_features).
         * @return Output tensor of shape (batch_size, out_features).
         */
        tensor forward(tensor &input) override;
        /**
         * @brief Returns trainable parameters of the layer.
         *
         * Typically, includes:
         * - weight matrix
         * - bias vector
         *
         * @return Vector of references to parameter tensors.
         */
        std::vector<_fw::ref<tensor>> parameters() override;
        /**
         * @brief Returns gradients corresponding to trainable parameters.
         *
         * The returned gradients must match the order of parameters().
         *
         * @return Vector of references to gradient tensors.
         */
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        tensor last_z;
        tensor weight, bias;
        _fw::i64 INPUT_FEATURES, OUTPUT_FEATURES;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP