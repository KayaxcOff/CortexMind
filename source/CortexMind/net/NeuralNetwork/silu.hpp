//
// Created by muham on 23.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_SILU_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_SILU_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class SiLU : public _fw::LayerBase {
    public:
        SiLU();
        ~SiLU() override;

        [[nodiscard]]
        tensor forward(const tensor &input) override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_SILU_HPP