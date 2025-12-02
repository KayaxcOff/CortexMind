//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/ActivationFunc/Tanh/tanh.hpp"
#include <cmath>

using namespace cortex::act;
using namespace cortex;

Tanh::Tanh() : cached_tanh(0, 0, 0) {}

Tanh::~Tanh() = default;

tensor Tanh::forward(const tensor &input) {
    const auto& output = input;

    for (size_t i = 0; i < output.get_data().size(); ++i) {
        output.get_data()[i] = std::tanh(output.get_data()[i]);
    }
    this->cached_tanh = output;
    return output;
}

tensor Tanh::backward(const tensor &grad_output) {
    const tensor tanh_sq = this->cached_tanh * this->cached_tanh;
    const tensor deriv = tanh_sq - 1.0;
    return deriv * grad_output;
}