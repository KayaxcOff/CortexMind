//
// Created by muham on 26.02.2026.
//

#include "CortexMind/net/NeuralNetwork/conv2d.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Conv2D::Conv2D(i64 in_channel, i64 out_channel, i64 kernel_width, i64 kernel_height, i64 stride, i64 padding, sys::device _dev) : Layer(true, "Conv2D") {
    this->kernel_ = std::make_unique<Kernel>(in_channel, out_channel, kernel_width, kernel_height, stride, padding, true, _dev);
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(tensor &input) {
    this->last_input = input;
    return this->kernel_->forward(this->last_input);
}

std::vector<ref<MindTensor> > Conv2D::parameters() {
    return {this->kernel_->getWeight(), this->kernel_->getBias()};
}

std::vector<ref<MindTensor> > Conv2D::gradients() {
    return {this->kernel_->getWeight().grad(), this->kernel_->getBias().grad()};
}