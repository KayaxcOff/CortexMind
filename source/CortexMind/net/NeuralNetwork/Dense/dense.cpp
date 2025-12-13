//
// Created by muham on 11.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Dense/dense.hpp"
#include <cmath>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Dense::Dense(const int in_size, const int out_size) : INPUT_SIZE(in_size), OUTPUT_SIZE(out_size) {
    this->weights.allocate(1, this->INPUT_SIZE, 1, this->OUTPUT_SIZE);
    this->biases.allocate(1, 1, 1, this->OUTPUT_SIZE);

    this->grad_biases.allocate(1, 1, 1, this->OUTPUT_SIZE);
    this->grad_weights.allocate(1, this->INPUT_SIZE, 1, this->OUTPUT_SIZE);

    const float limit = std::sqrt(1.0f / static_cast<float>(this->INPUT_SIZE));
    this->weights.uniform_rand(-limit, limit);

    this->biases.zero();
    this->grad_biases.zero();
    this->grad_weights.zero();
}

Dense::~Dense() = default;

tensor Dense::forward(const tensor &input) {
    this->input_cache = input;

    const tensor x = input.flatten();
    tensor output = x.matmul(this->weights) + this->biases;
    return output;
}

tensor Dense::backward(const tensor &grad_output) {
    const tensor& gradient = grad_output;

    const tensor x = this->input_cache.flatten();
    const tensor w = x.transpose().matmul(gradient);

    this->grad_weights += w;

    tensor b(1, 1, 1, this->OUTPUT_SIZE);

    const int batch = gradient.batch();

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->OUTPUT_SIZE; ++j) {
            b.at(0, 0, 0, j) += gradient.at(i, j, 0, 0);
        }
    }
    this->grad_biases += b;

    tensor grad_input = gradient.matmul(this->weights.transpose());
    return grad_input;
}

std::string Dense::config() {
    return "Dense";
}
