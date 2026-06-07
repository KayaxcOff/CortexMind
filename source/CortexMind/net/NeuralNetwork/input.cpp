//
// Created by muham on 22.05.2026.
//

#include "CortexMind/net/NeuralNetwork/input.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Input::Input(const std::span<int64> &_shape) : LayerBase("Input") {
    this->shape = _shape;
}

Input::~Input() = default;

tensor Input::forward(const tensor &input) {
    for (size_t i = 0; i < input.shape().size(); ++i) {
        CXM_ASSERT(this->shape[i] != input.shape()[i], "Shapes mismatch");
    }
    return input;
}

std::vector<ref<tensor>> Input::getParameters() {
    return {};
}

std::vector<ref<tensor>> Input::getGradients() {
    return {};
}