//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/NeuralNetwork/Flatten/flatten.hpp"

using namespace cortex::nn;
using namespace cortex;

tensor Flatten::forward(tensor &input) {
    this->originalShape = input._shape();

    const size_t B = this->originalShape[0];
    const size_t R = this->originalShape[1];
    const size_t C = this->originalShape[2];

    const size_t new_size = R * C;

    return input.reshape({B, 1, new_size});
}

tensor Flatten::backward(tensor &grad_output) {
    return grad_output.reshape(this->originalShape);
}

std::string Flatten::config() {
    return "Flatten";
}