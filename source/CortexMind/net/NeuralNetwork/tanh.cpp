//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/tanh.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Tanh::Tanh() : LayerBase("Tanh") {}

Tanh::~Tanh() = default;

tensor Tanh::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::tanh(input.get(), output.get(), input.len(), input.device());

    output.SetFlow(std::make_shared<meta::tanh>(input.pack(), output.pack()));

    return output;
}

std::vector<ref<tensor>> Tanh::getParameters() {
    return {};
}

std::vector<ref<tensor>> Tanh::getGradients() {
    return {};
}
