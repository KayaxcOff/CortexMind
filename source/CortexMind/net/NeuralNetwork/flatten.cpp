//
// Created by muham on 18.03.2026.
//

#include "CortexMind/net/NeuralNetwork/flatten.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Flatten::Flatten() : Layer(true, "Flatten") {}

Flatten::~Flatten() = default;

tensor Flatten::forward(tensor &input) {
    return input.flat();
}

std::vector<ref<tensor>> Flatten::parameters() {
    return {};
}

std::vector<ref<tensor>> Flatten::gradients() {
    return {};
}
