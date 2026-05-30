//
// Created by muham on 30.05.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_GLOBAL_AVG_POOL_2D_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_GLOBAL_AVG_POOL_2D_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class GlobalAveragePool2D : public _fw::LayerBase {
    public:
        GlobalAveragePool2D();
        ~GlobalAveragePool2D() override;

        [[nodiscard]]
        tensor forward(const tensor &input) override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getParameters() override;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> getGradients() override;
    };
} //namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_GLOBAL_AVG_POOL_2D_HPP