//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/NeuralNetwork/Conv2D/conv.hpp"

using namespace cortex::nn;
using namespace cortex;

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding) : input_cache(0, 0, 0) {
    this->mind_kernel_ = std::make_unique<fw::MindKernel>(in_channels, out_channels, kernel_size, stride, padding);
}

tensor Conv2D::forward(tensor &input) {
    this->input_cache = input;
    return this->mind_kernel_->apply(input);
}

tensor Conv2D::backward(tensor &grad_output) {
    return this->mind_kernel_->backward(this->input_cache, grad_output);
}

std::vector<tensor *> Conv2D::getParameters() {
    return {&this->mind_kernel_->get_weights(), &this->mind_kernel_->get_bias()};
}

std::vector<tensor *> Conv2D::getGradients() {
    return {&this->mind_kernel_->get_grad_weights(), &this->mind_kernel_->get_grad_bias()};
}

std::string Conv2D::config() {
    return "Conv2D";
}