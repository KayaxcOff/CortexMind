//
// Created by muham on 8.11.2025.
//

#include "CortexMind/Mind/NeuralNetwork/dense.hpp"

using namespace cortex::nn;

Dense::Dense(const size in, const size out) {
    this->weights.resize(in);
    this->biases.resize(out);
    this->gradWeights.resize(in);
}

Dense::~Dense() = default;

cortex::tensor Dense::forward(const tensor &input) {
    this->lastInput = input;

    for (size i = 0; i < this->weights.size(); i++) {
        // Perform matrix multiplication and add bias
        // Placeholder for actual implementation
    }
    return tensor(); // Placeholder return
}

cortex::tensor Dense::backward(const tensor &grad_output) {
    for (size i = 0; i < this->weights.size(); i++) {

    }
    this->outputGrad = grad_output;
    return tensor(); // Placeholder return
}

cortex::tensor Dense::getParams() const {
    return this->lastInput;
}

cortex::tensor Dense::getGrads() const {
    return this->outputGrad;
}