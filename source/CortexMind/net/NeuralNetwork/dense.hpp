//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/tools/device.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex::nn {
    class Dense : public _fw::LayerBase {
    public:
        Dense(int64 in, int64 out, _fw::sys::deviceType d_type = host);
        ~Dense() override;

        tensor forward(tensor &input) override;
        std::vector<tensor> getWeights() override;
        std::vector<tensor> getGradients() override;
    private:
        tensor kWeight;
        tensor kBias;

        int64 kInputDimension;
        int64 kOutputDimension;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_DENSE_HPP