//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/flatten.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Flatten::Flatten() : LayerBase("Flatten") {}

Flatten::~Flatten() = default;

tensor Flatten::forward(const tensor &input) {
    int64 batch_size = input.shape()[0];
    int64 flattened_size = static_cast<int64>(input.len()) / batch_size;

    return input.reshape({batch_size, flattened_size});
}

std::vector<ref<tensor>> Flatten::getParameters() {
    return {};
}

std::vector<ref<tensor>> Flatten::getGradients() {
    return {};
}