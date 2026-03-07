//
// Created by muham on 26.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class Flatten : _fw::Layer {
    public:
        Flatten();
        ~Flatten() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<_fw::MindTensor>> parameters() override;
        std::vector<_fw::ref<_fw::MindTensor>> gradients() override;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_FLATTEN_HPP