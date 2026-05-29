//
// Created by muham on 29.05.2026.
//

#include "CortexMind/net/NeuralNetwork/softmax.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#include <memory>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Softmax::Softmax() : LayerBase("Softmax") {}

Softmax::~Softmax() = default;

tensor Softmax::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::softmax(input.get(), output.get(), input.len(), input.device());

    if (output.has_grad()) [[likely]] {
        output.SetFlow(std::make_shared<meta::softmax>(input.pack(), output.pack()));
    }

    return output;
}

std::vector<ref<tensor>> Softmax::getParameters() {
    return {};
}

std::vector<ref<tensor>> Softmax::getGradients() {
    return {};
}