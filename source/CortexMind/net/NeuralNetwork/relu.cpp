//
// Created by muham on 26.02.2026.
//

#include "CortexMind/net/NeuralNetwork/relu.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

ReLU::ReLU() : Layer(true, "ReLU") {}

ReLU::~ReLU() = default;

tensor ReLU::forward(tensor &input) {
    this->last_input = input;
    this->last_input.clear_flow();
    return last_input.relu();
}

std::vector<ref<tensor>> ReLU::parameters() {
    return {};
}

std::vector<ref<tensor>> ReLU::gradients() {
    return {};
}