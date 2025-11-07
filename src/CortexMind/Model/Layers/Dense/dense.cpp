//
// Created by muham on 3.11.2025.
//


#include "CortexMind/Model/Layers/Dense/dense.hpp"

#include <CortexMind/Utils/MathTools/random_weight.hpp>

using namespace cortex::layer;
using namespace cortex;

Dense::Dense(const size_t in, const size_t out) : inputSize(in), outputSize(out), learning_rate(0.01) {

    weights.resize(outputSize, math::MindVector(inputSize));

    for (auto& val : weights) {
        for (auto& item : val) item = math::random_weight();
    }

    biases.resize(outputSize, 0.0);

    grad_weights.resize(outputSize, math::MindVector(inputSize, 0.0));
    grad_bias.resize(outputSize, 0.0);
}

Dense::~Dense() = default;

math::MindVector Dense::forward(const math::MindVector &input) {
    last_input = input;

    math::MindVector output (this->outputSize, 0.0);
    for (size_t i = 0; i < this->outputSize; ++i) {
        double sum = biases[i];
        for (size_t j = 0; j < this->inputSize; ++j) {
            if (input.size() != this->inputSize) {
                throw std::runtime_error("Dense::forward -> input size mismatch");
            }
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    return output;
}

math::MindVector Dense::backward(const math::MindVector& output_gradients) {
    for (auto& item : grad_weights) {
        std::ranges::fill(item, 0.0);
    }

    std::ranges::fill(grad_bias, 0.0);

    math::MindVector input_gradients (this->inputSize, 0.0);

    for (size_t i = 0; i < this->outputSize; ++i) {
        grad_bias[i] += output_gradients[i];
        for (size_t j = 0; j < this->inputSize; ++j) {
            if (output_gradients.size() != this->outputSize) {
                throw std::runtime_error("Dense::backward -> output_gradients size mismatch");
            }
            if (last_input.size() != this->inputSize) {
                throw std::runtime_error("Dense::backward -> last_input size mismatch");
            }

            grad_weights[i][j] += output_gradients[i] * last_input[j];
            input_gradients[j] += weights[i][j] * output_gradients[i];
        }
    }

    return input_gradients;
}

void Dense::update(const double lr) {
    for (size_t i = 0; i < this->outputSize; ++i) {
        for (size_t j = 0; j < this->inputSize; ++j) {
            this->weights[i][j] -= lr * this->grad_weights[i][j];
        }
        biases[i] -= lr * this->grad_bias[i];
    }
}

std::vector<math::MindVector*> Dense::get_parameters() {
    std::vector<math::MindVector*> params;
    for (auto& w : weights) params.push_back(&w);
    params.push_back(&biases);
    return params;
}

std::vector<math::MindVector*> Dense::get_gradients() {
    std::vector<math::MindVector*> grads;
    for (auto& gw : grad_weights) grads.push_back(&gw);
    grads.push_back(&grad_bias);
    return grads;
}