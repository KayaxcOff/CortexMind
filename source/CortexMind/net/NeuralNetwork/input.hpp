//
// Created by muham on 28.02.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Net/layer.hpp>
#include <vector>

namespace cortex::nn {
    class Input : _fw::Layer {
    public:
        explicit
        Input(const std::vector<int64>& shape);
        ~Input() override;

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        std::vector<int64> target_shape;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP