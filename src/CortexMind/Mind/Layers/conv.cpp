//
// Created by muham on 8.11.2025.
//

#include "CortexMind/Mind/NeuralNetwork/conv.hpp"

#include <CortexMind/Utils/MathTools/random.hpp>

using namespace cortex::nn;

Conv2D::Conv2D() : weights({}, {}), biases({}, {}), mind_kernel_(std::make_unique<tools::MindKernel>(random_seed())){}

Conv2D::~Conv2D() {
    this->mind_kernel_.reset();
}

cortex::tensor Conv2D::forward(const tensor &input) {
    for (auto output = this->mind_kernel_->apply(input); auto &row : output) {
        for (auto &val : row) {
            val += this->biases[0][0];
        }
    }

    return input;
}

cortex::tensor Conv2D::backward(const tensor &grad_output) {
    return grad_output;
}