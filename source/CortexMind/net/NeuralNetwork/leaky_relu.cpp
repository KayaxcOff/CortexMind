//
// Created by muham on 23.05.2026.
//

#include "CortexMind/net/NeuralNetwork/leaky_relu.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#include <string>
#include <memory>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

LeakyReLU::LeakyReLU(const float32 alpha) : LayerBase("LeakyReLU(" + std::to_string(alpha) + ")") {
    this->alpha = alpha;
}

LeakyReLU::~LeakyReLU() = default;

tensor LeakyReLU::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::leaky_relu(input.get(), output.get(), input.len(), input.device(), this->alpha);

    if (output.has_grad()) [[likely]] {
        output.SetFlow(std::make_shared<meta::leaky_relu>(input.pack(), this->alpha));
    }

    return output;
}

std::vector<ref<tensor>> LeakyReLU::getParameters() {
    return {};
}

std::vector<ref<tensor>> LeakyReLU::getGradients() {
    return {};
}

void LeakyReLU::Set(const float32 _alpha) {
    this->alpha = _alpha;
}