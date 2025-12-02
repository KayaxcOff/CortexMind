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

    tensor bias_expanded(output.get_shape()[0], 1, this->out_size);
    for (size_t b = 0; b < output.get_shape()[0]; b++) {
        for (size_t i = 0; i < this->out_size; i++) {
            bias_expanded(b,0,i) = this->bias(0,0,i);
        }
    }
    return output + bias_expanded;
}

tensor Dense::backward(tensor &grad_output) {
    this->gradWeights.zero();
    this->gradBias.zero();

    const size_t out = this->out_size;
    const size_t batch_size = grad_output.get_shape()[0];

    tensor grad_input(batch_size, 1, this->in_size);

    for (size_t i = 0; i < out; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < batch_size; ++j) {
            sum += grad_output(j, 0, i);
        }
        this->gradBias(0, 0, i) = sum;
    }

    tensor input = this->cached_input.transpose();

    tensor per_batch = input.matmul(grad_output);

    for (size_t i = 0; i < this->in_size; ++i) {
        for (size_t j = 0; j < this->out_size; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < batch_size; ++k) {
                sum += per_batch(k, i, j);
            }
            this->gradWeights(0, i, j) = sum;
        }
    }

    tensor weights_T = this->weights.transpose();
    tensor result = grad_output.matmul(weights_T);

    return result;
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