//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/convolution_2d.hpp"
#include <cmath>
#include <string>
#include <tuple>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Conv2D::Conv2D(int64 in_channels, int64 out_channels, int64 kH, int64 kW, const int64 sH, const int64 sW, const int64 pH, const int64 pW, const sys::DeviceType device) : LayerBase("Conv2D(" + std::to_string(out_channels) + ")"), KERNEL_WIDTH(kW), KERNEL_HEIGHT(kH) , STRIDE_WIDTH(sW), STRIDE_HEIGHT(sH) , PADDING_WIDTH(pW), PADDING_HEIGHT(pH) {
    this->weight = Tensor({out_channels, in_channels, kH, kW}, device, true);
    this->bias   = Tensor({out_channels}, device, true);

    const float64 limit = std::sqrt(6.0 / (static_cast<float64>(in_channels * kH * kW)));
    this->weight.uniform(-static_cast<float32>(limit), static_cast<float32>(limit));
    this->bias.zero();
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(const tensor &input) {
    CXM_ASSERT(input.ndim() != 4, "Conv2D expects 4D input (batch, channels, height, width)");

    const int64 batch_size   = input.shape()[0];
    const int64 in_channels  = input.shape()[1];
    const int64 in_height    = input.shape()[2];
    const int64 in_width     = input.shape()[3];

    CXM_ASSERT(in_channels != this->weight.shape()[1], "Input channels mismatch with Conv2D weight");

    auto [out_height, out_width] = compute_output_size(in_height, in_width);

    const Tensor col = this->im2col(input);

    const Tensor weight_flat = this->weight.reshape({this->weight.shape()[0], -1});

    Tensor output = weight_flat.matmul(col);

    output = output + this->bias.unsqueeze(1);

    output = output.reshape({batch_size, this->weight.shape()[0], out_height, out_width});

    return output;
}

std::vector<ref<tensor>> Conv2D::getParameters() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Conv2D::getGradients() {
    return {this->weight.grad(), this->bias.grad()};
}

tensor Conv2D::im2col(const tensor& input) const {
    CXM_ASSERT(input.ndim() != 4, "im2col expects 4D tensor");

    const auto [batch, channels, height, width] = std::tuple{
        input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]
    };

    const int64 out_h = (height + 2 * this->PADDING_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    const int64 out_w = (width  + 2 * this->PADDING_WIDTH  - this->KERNEL_WIDTH)  / this->STRIDE_WIDTH  + 1;

    const int64 kernel_elements = this->KERNEL_HEIGHT * this->KERNEL_WIDTH * channels;

    Tensor output({kernel_elements, batch * out_h * out_w}, input.device());

    return output;
}

std::pair<int64, int64> Conv2D::compute_output_size(const int64 input_height, const int64 input_width) const {
    int64 out_h = (input_height + 2 * this->PADDING_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    int64 out_w = (input_width  + 2 * this->PADDING_WIDTH  - this->KERNEL_WIDTH)  / this->STRIDE_WIDTH  + 1;

    CXM_ASSERT(out_h <= 0 || out_w <= 0, "Convolution output size became non-positive. Check kernel/stride/padding.");

    return {out_h, out_w};
}