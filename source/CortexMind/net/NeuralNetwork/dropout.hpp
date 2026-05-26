//
// Created by muham on 26.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DROPOUT_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DROPOUT_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Dropout layer - a powerful regularization technique.
     *
     * During training, randomly sets a fraction of input units to 0 at each update
     * to prevent overfitting. During evaluation, acts as an identity function.
     *
     * This implementation uses **inverted dropout** (scaling during training).
     */
    class Dropout : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs a Dropout layer.
         *
         * @param rate Dropout probability (fraction of units to drop).
         *             Should be in range [0.0, 1.0). Default: 0.1
         */
        explicit Dropout(float32 rate = 0.1f);
        ~Dropout() override;

        /**
         * @brief Performs forward pass with dropout.
         *
         * - In **training mode**: Applies dropout mask and scaling.
         * - In **evaluation mode**: Returns input unchanged (identity).
         *
         * @param input Input tensor
         * @return Output tensor after dropout (if training)
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * Dropout layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * Dropout layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    private:
        float32 rate;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DROPOUT_HPP