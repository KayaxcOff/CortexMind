//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/convolution_2d.hpp"
#include <string>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Conv2D::Conv2D(int64 in_channels, int64 out_channels, const int64 kH, const int64 kW, const int64 sH, const int64 sW, const int64 pH, const int64 pW, const sys::DeviceType device) : LayerBase("Conv2D(" + std::to_string(in_channels) + ", " + std::to_string(out_channels) + ")") {
    this->kernel = tensor({in_channels, out_channels}, device, true);
    this->bias = tensor({in_channels, out_channels}, device, true);

    this->KERNEL_HEIGHT = kH;
    this->KERNEL_WIDTH = kW;

    this->STRIDE_HEIGHT = sH;
    this->STRIDE_WIDTH = sW;

    this->PADDING_HEIGHT = pH;
    this->PADDING_WIDTH = pW;
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(const tensor &input) {
    return input;
}

std::vector<ref<tensor>> Conv2D::getParameters() {
    return {this->kernel, this->bias};
}

std::vector<ref<tensor>> Conv2D::getGradients() {
    return {this->kernel.grad(), this->bias.grad()};
}

tensor Conv2D::im2col() {
    return this->kernel;
}