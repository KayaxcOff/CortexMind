//
// Created by muham on 7.06.2026.
//

#include "CortexMind/net/NeuralNetwork/latent.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Latent::Latent() : LayerBase("Latent") {}

Latent::~Latent() = default;

tensor Latent::forward(const tensor &input) {
    this->cached_output = input.detach();
    return input;
}

std::vector<ref<tensor>> Latent::getParameters() {
    return {};
}

std::vector<ref<tensor>> Latent::getGradients() {
    return {};
}