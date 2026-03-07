//
// Created by muham on 28.02.2026.
//

#include "CortexMind/net/NeuralNetwork/input.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Input::Input(const std::vector<int64>& shape) : Layer(true, "Input"), target_shape(shape) {}

Input::~Input() = default;

tensor Input::forward(tensor &input) {
    this->last_input = input;
    CXM_ASSERT(input.shape() == this->target_shape, "cortex::nn::Input::forward()", "Shape mismatch");
    return this->last_input.detach();
}

std::vector<ref<tensor>> Input::parameters() {
    return {};
}

std::vector<ref<tensor>> Input::gradients() {
    return {};
}