//
// Created by muham on 7.06.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_LATENT_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_LATENT_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class Latent : public _fw::LayerBase {
    public:
        Latent();
        ~Latent() override;

        [[nodiscard]]
        tensor forward(const tensor &input) override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    private:
        tensor cached_output;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_LATENT_HPP