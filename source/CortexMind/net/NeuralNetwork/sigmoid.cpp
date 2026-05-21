//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/sigmoid.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Sigmoid::Sigmoid() : LayerBase("Sigmoid") {}

Sigmoid::~Sigmoid() = default;

tensor Sigmoid::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::sigmoid(input.get(), output.get(), input.len(), input.device());

    if (output.has_grad()) {
        output.SetFlow(std::make_shared<meta::sigmoid>(input.pack(), output.pack()));
    }

    return output;
}

std::vector<ref<tensor>> Sigmoid::getParameters() {
    return {};
}

std::vector<ref<tensor>> Sigmoid::getGradients() {
    return {};
}