//
// Created by muham on 28.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_GLOBAL_AVG_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_GLOBAL_AVG_HPP

#include <CortexMind/core/Net/layer.hpp>

namespace cortex::nn {
    class GlobalAvgPooling : public _fw::Layer {
    public:
        GlobalAvgPooling();
        ~GlobalAvgPooling() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_GLOBAL_AVG_HPP