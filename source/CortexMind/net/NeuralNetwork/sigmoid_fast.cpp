//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/sigmoid_fast.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

SigmoidFast::SigmoidFast() : LayerBase("SigmoidFast") {}

SigmoidFast::~SigmoidFast() = default;

tensor SigmoidFast::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::sigmoid_fast(input.get(), output.get(), input.len(), input.device());

    if (output.has_grad()) {
        output.SetFlow(std::make_shared<meta::sigmoid>(input.pack(), output.pack()));
    }

    return output;
}

std::vector<ref<tensor>> SigmoidFast::getParameters() {
    return {};
}

std::vector<ref<tensor>> SigmoidFast::getGradients() {
    return {};
}
