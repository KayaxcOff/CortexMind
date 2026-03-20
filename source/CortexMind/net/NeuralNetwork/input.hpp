//
// Created by muham on 19.03.2026.
//

#ifndef CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP
#define CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <initializer_list>
#include <vector>

namespace cortex::nn {
    class Input : public _fw::Layer {
    public:
        explicit Input(const std::vector<int64>& shape);
        Input(std::initializer_list<int64> shape);

        tensor forward(tensor &input) override;
        std::vector<_fw::ref<tensor>> parameters() override;
        std::vector<_fw::ref<tensor>> gradients() override;
    private:
        std::vector<int64> _shape;
    };
} // namespace cortex::nn

#endif //CORTEXMIND_NET_NEURAL_NETWORK_INPUT_HPP