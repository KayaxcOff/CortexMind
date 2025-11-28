//
// Created by muham on 9.11.2025.
//

#include "CortexMind/Mind/ActivationFunc/relu.hpp"

using namespace cortex::act;

ReLU::ReLU() : cached_input(0, 0) {}

ReLU::~ReLU() = default;

cortex::tensor ReLU::forward(const tensor &input) {
    this->cached_input = input;

    tensor output(input.get_rows(), input.get_cols());

    for (size i = 0; i < input.get_rows(); ++i) {
        for (size j = 0; j < input.get_cols(); ++j) {
            const float64 value = input(i, j);
            output(i, j) = value > 0 ? value : 0;
        }
    }
    return output;
}

cortex::tensor ReLU::backward(const tensor &grad_output) {
    if (this->cached_input.get_rows() == 0 && this->cached_input.get_cols() == 0) {
        throw std::runtime_error("ReLU backward called before forward.");
    }

    if (grad_output.get_cols() != this->cached_input.get_cols() || grad_output.get_rows() != this->cached_input.get_rows()) {
        throw std::invalid_argument("Gradient output shape does not match cached input shape.");
    }

    tensor grad_input(grad_output.get_rows(), grad_output.get_cols());

    for (size i = 0; i < grad_output.get_rows(); ++i) {
        for (size j = 0; j < grad_output.get_cols(); ++j) {
            grad_input(i, j) = this->cached_input(i, j) > 0.0 ? grad_output(i, j) : 0.0;
        }
    }

    return grad_input;
}