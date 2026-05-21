//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class ReLU : public _fw::LayerBase {
    public:
        ReLU();
        ~ReLU() override;

        [[nodiscard]]
        tensor forward(const tensor &input) override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP