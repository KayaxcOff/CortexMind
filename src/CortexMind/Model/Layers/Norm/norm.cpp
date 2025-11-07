//
// Created by muham on 3.11.2025.
//

#include "CortexMind/Model/Layers/Norm/norm.hpp"

using namespace cortex::layer;
using namespace cortex;

Normalize::Normalize(const std::vector<int>& _input) {
    this->input = _input;
}

Normalize::~Normalize() = default;

math::MindVector Normalize::forward(const math::MindVector& input) {
    math::MindVector output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        constexpr double scale = 255.0;
        output[i] = input[i] / scale;
    }
    return output;
}

math::MindVector Normalize::backward(const math::MindVector& grad_output) {
    math::MindVector grad_input(grad_output.size());

    for (size_t i = 0; i < grad_output.size(); ++i) {
        constexpr double scale = 255.0;
        grad_input[i] = grad_output[i] / scale;
    }
    return grad_input;
}

void Normalize::update(double lr) {
    // No parameters to update in normalization layer
}

std::vector<math::MindVector*> Normalize::get_parameters() {
    return {};
}

std::vector<math::MindVector*> Normalize::get_gradients() {
    return {};
}