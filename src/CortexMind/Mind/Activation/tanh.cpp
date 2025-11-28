//
// Created by muham on 28.11.2025.
//

#include "CortexMind/Mind/ActivationFunc/tanh.hpp"

#include <cmath>

using namespace cortex::act;

Tanh::Tanh() : cached_input(0, 0) {}

Tanh::~Tanh() = default;

cortex::tensor Tanh::forward(const tensor &input) {
    this->cached_input = tensor(input.get_rows(), input.get_cols());

    auto output = tensor(input.get_rows(), input.get_cols());

    for (size i = 0; i < input.get_rows(); ++i) {
        for (size j = 0; j < input.get_cols(); ++j) {
            const float64 x = input(i, j);
            const float64 tanh_x = std::tanh(x);

            output(i, j) = tanh_x;
            this->cached_input(i, j) = tanh_x;
        }
    }

    return output;
}

cortex::tensor Tanh::backward(const tensor &grad_output) {
    if (this->cached_input.get_rows() == 0 || this->cached_input.get_cols() == 0) {
        throw std::runtime_error("Tanh backward called before forward.");
    }

    if (this->cached_input.get_rows() != grad_output.get_rows() || this->cached_input.get_cols() != grad_output.get_cols()) {
        throw std::runtime_error("Tanh backward input size mismatch.");
    }

    tensor grad_input(grad_output.get_rows(), grad_output.get_cols());

    for (size i = 0; i < grad_output.get_rows(); ++i) {
        for (size j = 0; j < grad_output.get_cols(); ++j) {
            const float64 x = grad_output(i, j);
            const float64 dt = 1.0 - (x * x);

            grad_input(i, j) = dt * grad_output(i, j);
        }
    }

    return grad_input;
}
