//
// Created by muham on 22.05.2026.
//

#include "CortexMind/net/NeuralNetwork/input.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Input::Input(const std::vector<int64> &_shape) : LayerBase("Input"), shape(_shape) {}

Input::~Input() = default;

tensor Input::forward(const tensor &input) {
    CXM_ASSERT(this->shape != input.shape(), "Input tensor shape mismatch");
    return input;
}

std::vector<ref<tensor>> Input::getParameters() {
    return {};
}

std::vector<ref<tensor>> Input::getGradients() {
    return {};
}