//
// Created by muham on 7.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Flatten/flatten.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Flatten::Flatten(std::unique_ptr<ActivationFunc> activation_func) : Layer(std::move(activation_func)), originalShape({}) {}
Flatten::~Flatten() = default;

tensor Flatten::forward(const tensor &input) {
    this->originalShape = input.shape();

    return input.flatten();
}

tensor Flatten::backward(const tensor &grad_output) {
    return grad_output.flatten();
}

std::string Flatten::config() const {
    return "Flatten";
}

std::vector<std::reference_wrapper<tensor>> Flatten::gradients() {
    return {};
}

std::vector<std::reference_wrapper<tensor>> Flatten::parameters() {
    return {};
}