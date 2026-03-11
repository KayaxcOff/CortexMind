//
// Created by muham on 25.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <CortexMind/core/Engine/Memory/device.hpp>

namespace cortex::nn {
    class Dense : public _fw::Layer {
    public:

        Dense(int64 in_feats, int64 out_feats, _fw::sys::device _dev = _fw::sys::device::host);
        ~Dense() override;

        tensor forward(tensor &input) override;

        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        tensor last_z;
        tensor weight, bias;
        int64 INPUT_FEATURES, OUTPUT_FEATURES;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP