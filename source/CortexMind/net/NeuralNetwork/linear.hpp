//
// Created by muham on 25.04.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_LINEAR_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_LINEAR_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Memory/device.hpp>

namespace cortex::nn {
    class Linear : public _fw::LayerBase {
    public:
        Linear(int64 input_dim, int64 output_dim, _fw::sys::deviceType d_type = _fw::sys::host);
        ~Linear() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> getWeight() override;
        std::vector<_fw::ref<tensor>> getGradient() override;
    private:
        tensor weight;
        tensor bias;

        int64 input_dim;
        int64 output_dim;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_LINEAR_HPP