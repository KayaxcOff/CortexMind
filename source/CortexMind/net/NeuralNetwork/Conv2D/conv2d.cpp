//
// Created by muham on 11.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Conv2D/conv2d.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Conv2D::Conv2D(int in_channel, int out_channel, int kernel_width, int kernel_height) {
    this->mind_kernel_ = std::make_unique<MindKernel>(in_channel, out_channel, kernel_width, kernel_height);
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(const tensor &input) {
    this->input_cache = input;
    return this->mind_kernel_->apply(input);
}

tensor Conv2D::backward(const tensor &grad_output) {
    return this->mind_kernel_->backward(this->input_cache, grad_output);
}

std::string Conv2D::config() {
    return "Conv2D";
}

std::array<tensor *, 2> Conv2D::parameters() {
    return {&this->weights, &this->biases};
}

std::array<tensor *, 2> Conv2D::gradients() {
    return {&this->grad_biases, &this->grad_weights};
}