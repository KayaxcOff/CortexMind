//
// Created by muham on 28.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_TANH_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_TANH_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class Tanh : public _fw::Layer {
    public:
        Tanh();
        ~Tanh() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_TANH_HPP