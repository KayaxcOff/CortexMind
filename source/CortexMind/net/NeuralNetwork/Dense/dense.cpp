//
// Created by muham on 11.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Dense/dense.hpp"
#include <cmath>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Dense::Dense(const int in_size, const int out_size) : INPUT_SIZE(in_size), OUTPUT_SIZE(out_size) {
    this->weights.allocate(1, 1, this->INPUT_SIZE, this->OUTPUT_SIZE);
    this->biases.allocate(1, 1, 1, this->OUTPUT_SIZE);

    this->grad_weights.allocate(1, 1, this->INPUT_SIZE, this->OUTPUT_SIZE);
    this->grad_biases.allocate(1, 1, 1, this->OUTPUT_SIZE);

    const float limit = std::sqrt(1.0f / static_cast<float>(this->INPUT_SIZE));
    this->weights.uniform_rand(-limit, limit);
    this->biases.zero();
    this->grad_weights.zero();
    this->grad_biases.zero();
}

Dense::~Dense() = default;

tensor Dense::forward(const tensor &input) {
    this->input_cache = input;

    tensor x_flat = input.flatten();

    tensor bias_expanded(input.batch(), 1, 1, this->OUTPUT_SIZE);
    for (int i = 0; i < input.batch(); ++i) {
        for (int j = 0; j < this->OUTPUT_SIZE; ++j) {
            bias_expanded.at(i,0,0,j) = this->biases.at(0,0,0,j);
        }
    }

    tensor y = input.matmul(this->weights) + bias_expanded;

    return y;
}

tensor Dense::backward(const tensor &grad_output) {
    const tensor &gradient = grad_output;
    const int batch = gradient.batch();


    tensor x_flat(batch, 1, 1, this->INPUT_SIZE);
    for (int i = 0; i < batch; ++i)
        for (int j = 0; j < this->INPUT_SIZE; ++j)
            x_flat.at(i,0,0,j) = this->input_cache.at(i,0,0,j);

    tensor grad_flat(batch, 1, 1, this->OUTPUT_SIZE);
    for (int i = 0; i < batch; ++i)
        for (int j = 0; j < this->OUTPUT_SIZE; ++j)
            grad_flat.at(i,0,0,j) = gradient.at(i,0,0,j);

    tensor dw = x_flat.transpose().matmul(grad_flat);

    tensor dw_tensor(1,1,this->INPUT_SIZE,this->OUTPUT_SIZE);
    for (int i = 0; i < this->INPUT_SIZE; ++i)
        for (int j = 0; j < this->OUTPUT_SIZE; ++j)
            dw_tensor.at(0,0,i,j) = dw.at(i,0,0,j);

    this->grad_weights += dw_tensor;

    tensor db(1,1,1,this->OUTPUT_SIZE);
    for (int j = 0; j < this->OUTPUT_SIZE; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < batch; ++i) {
            sum += gradient.at(i,0,0,j);
        }
        db.at(0,0,0,j) = sum;
    }
    this->grad_biases += db;

    tensor grad_input = grad_flat.matmul(this->weights.transpose());

    return grad_input;
}

std::string Dense::config() {
    return "Dense";
}

void Dense::register_params(Optimizer &optim_fn) {
    if (!this->weights.empty()) {
        optim_fn.add_param(&this->weights, &this->grad_biases);
    }
    if (!this->biases.empty()) {
        optim_fn.add_param(&this->biases, &this->grad_biases);
    }
}
