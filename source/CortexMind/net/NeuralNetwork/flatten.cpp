//
// Created by muham on 26.02.2026.
//

#include "CortexMind/net/NeuralNetwork/flatten.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Flatten::Flatten() : Layer(true, "Flatten") {}

Flatten::~Flatten() = default;

tensor Flatten::forward(tensor &input) {
    this->last_input = input;
    return last_input.flatten();
}

std::vector<ref<tensor>> Flatten::parameters() {
    return {};
}

std::vector<ref<tensor>> Flatten::gradients() {
    return {};
}