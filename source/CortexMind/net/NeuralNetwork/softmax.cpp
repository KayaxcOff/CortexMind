//
// Created by muham on 8.03.2026.
//

#include "CortexMind/net/NeuralNetwork/softmax.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Softmax::Softmax() : Layer(true, "Softmax") {}

Softmax::~Softmax() = default;

tensor Softmax::forward(tensor &input) {
    this->last_input = input;
    this->last_input.clear_flow();

    const float32 max_val = this->last_input.max();
    const tensor shifted = this->last_input - max_val;

    const tensor ex = shifted.exp();
    const tensor sum = ex.sum();

    return ex / sum;
}

std::vector<ref<tensor>> Softmax::parameters() {
    return {};
}

std::vector<ref<tensor>> Softmax::gradients() {
    return {};
}
