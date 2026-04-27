//
// Created by muham on 25.04.2026.
//

#include "CortexMind/net/NeuralNetwork/flatten.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Flatten::Flatten() : LayerBase("Flatten") {}

Flatten::~Flatten() = default;

tensor Flatten::forward(tensor &input) {
    if (input.ndim() == 1) {
        return input;
    }

    int64 batch = input.shape()[0];
    int64 feat = 1;

    for (size_t i = 1; i < input.shape().size(); ++i) {
        feat *= input.shape()[i];
    }

    auto output = tensor({batch, feat}, input.get(), input.device(), input.isGradRequired());

    return output;
}

std::vector<ref<tensor>> Flatten::getWeight() {
    return {};
}

std::vector<ref<tensor>> Flatten::getGradient() {
    return {};
}
