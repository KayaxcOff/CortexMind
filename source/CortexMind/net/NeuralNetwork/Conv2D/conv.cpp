//
// Created by muham on 7.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Conv2D/conv.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Conv2D::Conv2D(std::unique_ptr<ActivationFunc> activation_func, int in_channels, int out_channels, int kernel_height , int kernel_width) : Layer(std::move(activation_func)), KERNEL_HEIGHT(kernel_height), KERNEL_WIDTH(kernel_width) {
    this->conv_kernel_ = std::make_unique<ConvKernel>(in_channels, out_channels, kernel_height, kernel_width);
}

Conv2D::~Conv2D() {
    this->conv_kernel_.reset();
}

tensor Conv2D::forward(const tensor &input) {
    this->input_cache = input;
    return this->conv_kernel_->apply(input);
}

tensor Conv2D::backward(const tensor &grad_output) {
    return this->conv_kernel_->backward(this->input_cache, grad_output);
}

std::string Conv2D::config() const {
    return "Conv2D";
}

std::vector<std::reference_wrapper<tensor>> Conv2D::gradients() {
    return {};
}

std::vector<std::reference_wrapper<tensor>> Conv2D::parameters() {
    return {};
}