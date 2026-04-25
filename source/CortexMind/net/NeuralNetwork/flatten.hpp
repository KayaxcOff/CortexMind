//
// Created by muham on 25.04.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    /**
     * @brief Flatten layer that reshapes multi-dimensional input into 2D.
     *
     * Keeps the first dimension (usually batch size) unchanged and flattens
     * all remaining dimensions into a single feature dimension.
     *
     * Example: [batch, C, H, W] → [batch, C*H*W]
     */
    class Flatten : public _fw::LayerBase {
    public:
        Flatten();
        ~Flatten() override;

        /**
         * @brief Flattens the input tensor to 2D while preserving the batch dimension.
         *
         * The first dimension is treated as batch size and kept as-is.
         * All subsequent dimensions are multiplied together to form the feature size.
         *
         * @param input Input tensor of any shape (minimum 2D)
         * @return 2D output tensor with shape [batch_size, features]
         */
        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> getWeight() override;
        std::vector<_fw::ref<tensor>> getGradient() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP