//
// Created by muham on 26.02.2026.
//

#include "CortexMind/net/NeuralNetwork/conv2d.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Conv2D::Conv2D(int64 in_channel, int64 out_channel, int64 kernel_width, int64 kernel_height, int64 stride, int64 padding, sys::device _dev) : Layer(true, "Conv2D") {
    this->kernel_ = std::make_unique<Kernel>(in_channel, out_channel, kernel_width, kernel_height, stride, padding, true, _dev);
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(tensor &input) {
    this->last_input = input;
    this->last_input.clear_flow();
    return this->kernel_->forward(this->last_input);
}

std::vector<ref<MindTensor> > Conv2D::parameters() {
    return {this->kernel_->getWeight(), this->kernel_->getBias()};
}

std::vector<ref<MindTensor> > Conv2D::gradients() {
    return {this->kernel_->getWeight().grad(), this->kernel_->getBias().grad()};
}