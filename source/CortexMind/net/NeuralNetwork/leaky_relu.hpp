//
// Created by muham on 6.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex::nn {
    class LeakyReLU : public _fw::Layer {
    public:
        explicit
        LeakyReLU(float32 alpha = 0.01f);
        ~LeakyReLU() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<_fw::MindTensor>> parameters() override;
        std::vector<_fw::ref<_fw::MindTensor>> gradients() override;
    private:
        float32 alpha;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_LEAKY_RELU_HPP