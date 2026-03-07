//
// Created by muham on 26.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class ReLU : public _fw::Layer {
    public:
        ReLU();
        ~ReLU() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_RELU_HPP