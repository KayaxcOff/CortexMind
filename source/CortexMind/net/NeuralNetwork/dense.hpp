//
// Created by muham on 3.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class Dense : public _fw::LayerBase {
    public:
        Dense(int64 in_dim, int64 out_dim, _fw::sys::deviceType d_type = _fw::sys::host);
        ~Dense() override;

        [[nodiscard]]
        tensor forward(tensor &input) override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getWeight() override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradient() override;
    private:
        tensor weight;
        tensor bias;

        int64 INPUT_DIM, OUTPUT_DIM;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP