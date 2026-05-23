//
// Created by muham on 23.05.2026.
//

#include "CortexMind/net/NeuralNetwork/silu_fast.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

SiLUFast::SiLUFast() : LayerBase("SiLUFast") {}

SiLUFast::~SiLUFast() = default;

tensor SiLUFast::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::silu_fast(input.get(), output.get(), input.len(), input.device());

    return output;
}

std::vector<ref<tensor>> SiLUFast::getParameters() {
    return {};
}

std::vector<ref<tensor>> SiLUFast::getGradients() {
    return {};
}