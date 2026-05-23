//
// Created by muham on 23.05.2026.
//

#include "CortexMind/net/NeuralNetwork/silu.hpp"
#include <CortexMind/framework/Engine/IX/activation.hpp>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

SiLU::SiLU() : LayerBase("SiLU") {}

SiLU::~SiLU() = default;

tensor SiLU::forward(const tensor &input) {
    tensor output(input.shape(), input.device(), input.has_grad());

    ix::Activation::silu(input.get(), output.get(), input.len(), input.device());

    return output;
}

std::vector<ref<tensor>> SiLU::getParameters() {
    return {};
}

std::vector<ref<tensor>> SiLU::getGradients() {
    return {};
}