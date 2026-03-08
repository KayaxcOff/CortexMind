//
// Created by muham on 8.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SOFTMAX_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SOFTMAX_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class Softmax : public _fw::Layer {
    public:
        Softmax();
        ~Softmax() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SOFTMAX_HPP