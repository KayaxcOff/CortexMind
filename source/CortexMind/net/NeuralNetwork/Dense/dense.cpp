//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/NeuralNetwork/Dense/dense.hpp"
#include <cmath>

using namespace cortex::nn;
using namespace cortex;

Dense::Dense(const size_t in, const size_t out) : weights(1, in, out, true), bias(1, 1, out, true), gradWeights(1, in, out), gradBias(1, 1, out), cached_input(0,0,0), in_size(in), out_size(out) {
    const double limit = std::sqrt(1.0 / static_cast<double>(in_size));

    this->weights.uniform_rand(-limit, limit);
    this->bias.fill(0.0);
}

tensor Dense::forward(tensor &input) {
    this->cached_input = input;
    tensor output = input.matmul(this->weights);

    for (size_t b = 0; b < output.get_shape()[0]; ++b) {
        for (size_t j = 0; j < this->out_size; ++j) {
            output(b, 0, j) += this->bias(0, 0, j);
        }
    }
    return output;
}

tensor Dense::backward(tensor &grad_output) {
    const size_t batch_size = grad_output.get_shape()[0];

    for (size_t j = 0; j < this->out_size; ++j) {
        double sum = 0.0;
        for (size_t b = 0; b < batch_size; ++b) {
            sum += grad_output(b, 0, j);
        }
        this->gradBias(0, 0, j) = sum;
    }

    for (size_t i = 0; i < this->in_size; ++i) {
        for (size_t j = 0; j < this->out_size; ++j) {
            double sum = 0.0;
            for (size_t b = 0; b < batch_size; ++b) {
                sum += this->cached_input(b, 0, i) * grad_output(b, 0, j);
            }
            this->gradWeights(0, i, j) = sum;
        }
    }

    tensor grad_input(batch_size, 1, this->in_size);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < this->in_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < this->out_size; ++j) {
                sum += grad_output(b, 0, j) * this->weights(0, i, j);
            }
            grad_input(b, 0, i) = sum;
        }
    }

    return grad_input;
}

std::vector<tensor*> Dense::getGradients() {
    return {&this->gradWeights, &this->gradBias};
}

std::vector<tensor*> Dense::getParameters() {
    return {&this->weights, &this->bias};
}

std::string Dense::config() {
    return "Dense";
}