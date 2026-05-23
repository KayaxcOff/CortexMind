//
// Created by muham on 23.05.2026.
//

#include "CortexMind/net/NeuralNetwork/gelu_exact.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

GeLUExact::GeLUExact() : LayerBase("GeLUExact") {}

GeLUExact::~GeLUExact() = default;

tensor GeLUExact::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::gelu_exact(input.get(), output.get(), input.len(), input.device());

    if (output.has_grad()) [[likely]] {
        output.SetFlow(std::make_shared<meta::gelu_exact>(input.pack(), output.pack()));
    }

    return output;
}

std::vector<ref<tensor>> GeLUExact::getParameters() {
    return {};
}

std::vector<ref<tensor>> GeLUExact::getGradients() {
    return {};
}