//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/NeuralNetwork/Flatten/flatten.hpp"

using namespace cortex::nn;
using namespace cortex;

Flatten::Flatten() : reshapedInput(0, 0, 0) {}

tensor Flatten::forward(tensor &input) {
    this->originalShape = input.get_shape();

    const size_t B = this->originalShape[0];
    const size_t R = this->originalShape[1];
    const size_t C = this->originalShape[2];

    const size_t new_size = R * C;

    this->reshapedInput = input.reshape({B, 1, new_size});

    return this->reshapedInput;
}

tensor Flatten::backward(tensor &grad_output) {
    return grad_output.reshape(this->originalShape);
}

std::string Flatten::config() {
    return "Flatten";
}