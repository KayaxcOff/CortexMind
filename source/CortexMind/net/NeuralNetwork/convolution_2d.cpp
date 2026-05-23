//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/convolution_2d.hpp"
#include <cmath>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Conv2D::Conv2D(int64 in_channels, int64 out_channels, int64 kH, int64 kW, const int64 sH, const int64 sW, const int64 pH, const int64 pW, const sys::DeviceType device) : LayerBase("Conv2D(" + std::to_string(out_channels) + ")"), KERNEL_WIDTH(kW), KERNEL_HEIGHT(kH) , STRIDE_WIDTH(sW), STRIDE_HEIGHT(sH) , PADDING_WIDTH(pW), PADDING_HEIGHT(pH) {this->weight = Tensor({out_channels, in_channels, kH, kW}, device, true);
    this->bias   = Tensor({out_channels}, device, true);

    const float64 limit = std::sqrt(6.0 / (static_cast<float64>(in_channels * kH * kW)));
    this->weight.uniform(-static_cast<float32>(limit), static_cast<float32>(limit));
    this->bias.zero();
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(const tensor &input) {
    // I'll write Convolution class in ix namespace
    return input;
}

std::vector<ref<tensor>> Conv2D::getParameters() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Conv2D::getGradients() {
    return {this->weight.grad(), this->bias.grad()};
}