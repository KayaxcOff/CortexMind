//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Flatten layer.
     *
     * Reshapes the input tensor while preserving the batch dimension.
     * Typically used before fully connected (Dense) layers to convert
     * feature maps (e.g. from Conv2D) into a 1D feature vector per sample.
     *
     * Example: Input shape `(batch, C, H, W)` → Output shape `(batch, C*H*W)`
     */
    class Flatten : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Flatten layer.
         */
        Flatten();
        ~Flatten() override;

        /**
         * @brief Performs the forward pass by flattening the input tensor.
         *
         * Keeps the batch dimension unchanged and flattens all remaining dimensions
         * into a single feature dimension.
         *
         * @param input Input tensor (any shape with at least 2 dimensions)
         * @return Flattened tensor of shape `(batch_size, features)`
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * Flatten layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * Flatten layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP