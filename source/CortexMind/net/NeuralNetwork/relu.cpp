//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/relu.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

ReLU::ReLU() : LayerBase("ReLU") {}

ReLU::~ReLU() = default;

tensor ReLU::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::relu(input.get(), output.get(), input.len(), input.device());

    output.SetFlow(std::make_shared<meta::relu>(input.pack()));

    return output;
}

std::vector<ref<tensor>> ReLU::getParameters() {
    return {};
}

std::vector<ref<tensor>> ReLU::getGradients() {
    return {};
}