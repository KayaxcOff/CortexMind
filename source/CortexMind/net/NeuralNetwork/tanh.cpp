//
// Created by muham on 28.02.2026.
//

#include "CortexMind/net/NeuralNetwork/tanh.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Tanh::Tanh() : Layer(true, "Tanh") {}

Tanh::~Tanh() = default;

tensor Tanh::forward(tensor &input) {
    this->last_input = input;
    return this->last_input.tanh();
}

std::vector<ref<tensor> > Tanh::parameters() {
    return {};
}

std::vector<ref<tensor> > Tanh::gradients() {
    return {};
}