//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP

#include <CortexMind/tools/device.hpp>
#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief   Fully connected (dense) layer: linear transformation + bias
     *
     * y = x @ W^T + b
     */
    class Dense : public _fw::Layer {
    public:
        /**
         * @brief   Constructs a dense layer
         * @param   in_feats    Number of input features
         * @param   out_feats   Number of output features
         * @param   d           Target device (host or cuda)
         *
         * @note    Weights initialized with Kaiming uniform:
         *          limit = √(6 / (in_features + out_features))
         *          W ~ Uniform(-limit, limit)
         * @note    Bias initialized to zero
         * @note    Both weight and bias have requires_grad = true
         */
        Dense(int64 in_feats, int64 out_feats, dev d = dev::host);
        ~Dense() override;

        /**
         * @brief   Forward pass: computes linear transformation + bias
         * @param   input   Input tensor (shape: [batch, ..., in_features])
         * @return  Output tensor (shape: [batch, ..., out_features])
         *
         * @note    Uses broadcasting for bias addition
         * @note    Dispatches to device-specific matmul (AVX2 or cuBLAS)
         */
        [[nodiscard]]
        tensor forward(tensor &input) override;
        /**
         * @brief   Returns non-owning references to trainable parameters
         * @return  {weight, bias}
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> parameters() override;
        /**
         * @brief   Returns non-owning references to parameter gradients
         * @return  {weight.grad(), bias.grad()}
         *
         * @note    Gradients exist only if requires_grad = true
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        tensor weight, bias;
        int64 INPUT_FEATS, OUTPUT_FEATS;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP