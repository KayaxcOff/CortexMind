//
// Created by muham on 22.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <vector>

namespace cortex::nn {
    /**
     * @brief Input layer.
     *
     * This layer serves as the entry point of the neural network. It primarily
     * performs input shape validation and acts as an identity transformation.
     *
     * It is useful for:
     * - Explicitly defining the expected input shape of the model
     * - Improving model readability and documentation
     * - Enforcing shape consistency at the beginning of the forward pass
     */
    class Input : public _fw::LayerBase {
    public:
        /**
         * @brief Constructs an Input layer with expected shape.
         *
         * @param _shape Expected shape of the input tensor (e.g. {batch_size, features})
         */
        explicit Input(const std::vector<int64>& _shape);
        ~Input() override;

        /**
         * @brief Performs forward pass (identity operation with shape check).
         *
         * Validates that the input tensor matches the expected shape and
         * returns the input tensor unchanged.
         *
         * @param input Input tensor
         * @return The same input tensor (identity)
         */
        [[nodiscard]]
        tensor forward(const tensor &input) override;
        /**
         * @brief Returns the trainable parameters of this layer.
         *
         * Input layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        /**
         * @brief Returns the gradients of the trainable parameters.
         *
         * Input layer has no trainable parameters.
         *
         * @return Empty vector
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    private:
        std::vector<int64> shape;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP