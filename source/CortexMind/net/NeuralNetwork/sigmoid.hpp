//
// Created by muham on 1.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class Sigmoid : public _fw::Layer {
    public:
        Sigmoid();
        ~Sigmoid() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<_fw::MindTensor>> parameters() override;
        std::vector<_fw::ref<_fw::MindTensor>> gradients() override;
    private:
        tensor last_output;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SIGMOID_HPP