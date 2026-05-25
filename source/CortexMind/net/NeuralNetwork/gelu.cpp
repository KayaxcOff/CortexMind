//
// Created by muham on 23.05.2026.
//

#include "CortexMind/net/NeuralNetwork/gelu.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#include <memory>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

GeLU::GeLU() : LayerBase("GeLU") {}

GeLU::~GeLU() = default;

tensor GeLU::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::gelu(input.get(), output.get(), input.len(), input.device());

    if (output.has_grad()) [[likely]] {
        output.SetFlow(std::make_shared<meta::gelu>(input.pack(), output.pack()));
    }

    return output;
}

std::vector<ref<tensor>> GeLU::getParameters() {
    return {};
}

std::vector<ref<tensor>> GeLU::getGradients() {
    return {};
}