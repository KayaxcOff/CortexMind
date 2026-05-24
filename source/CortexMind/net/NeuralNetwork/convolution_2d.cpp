//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/NeuralNetwork/convolution_2d.hpp"
#include <CortexMind/framework/Engine/IX/convolution.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#include <cmath>
#include <iostream>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Conv2D::Conv2D(int64 in_channels, int64 out_channels, int64 kH, int64 kW, const int64 sH, const int64 sW, const int64 pH, const int64 pW, const sys::DeviceType device) : LayerBase("Conv2D(" + std::to_string(out_channels) + ")") {
    this->weight = Tensor({out_channels, in_channels, kH, kW}, device, true);
    this->bias = Tensor({out_channels}, device, true);

    this->STRIDE_HEIGHT = sH;
    this->STRIDE_WIDTH = sW;

    this->PADDING_HEIGHT = pH;
    this->PADDING_WIDTH = pW;

    this->KERNEL_HEIGHT = kH;
    this->KERNEL_WIDTH = kW;

    const float64 limit = std::sqrt(6.0 / (static_cast<float64>(in_channels * kH * kW)));
    this->weight.uniform(-static_cast<float32>(limit), static_cast<float32>(limit));
    this->bias.zero();
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(const tensor& input) {
    CXM_ASSERT(input.ndim() != 4, "Conv2D expects 4D input (N,C,H,W)");
    CXM_ASSERT(input.shape()[1] != this->weight.shape()[1], "Input channels mismatch");

    const i64 N  = input.shape()[0];
    const i64 H  = input.shape()[2];
    const i64 W  = input.shape()[3];
    const i64 oC = this->weight.shape()[0];

    auto [oH, oW] = compute_output_size(H, W);

    const Tensor col = this->im2col(input);

    const Tensor weight_flat = this->weight.reshape({oC, -1});

    Tensor output = weight_flat.matmul(col);

    output = output + this->bias.reshape({oC, 1});

    output = output.reshape({oC, N, oH, oW});
    output = output.permute({1,0,2,3});

    return output;
}

std::vector<ref<tensor>> Conv2D::getParameters() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Conv2D::getGradients() {
    return {this->weight.grad(), this->bias.grad()};
}

std::pair<int64, int64> Conv2D::compute_output_size(const int64 input_height, const int64 input_width) const {
    int64 out_h = (input_height + 2 * this->PADDING_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    int64 out_w = (input_width  + 2 * this->PADDING_WIDTH  - this->KERNEL_WIDTH)  / this->STRIDE_WIDTH  + 1;

    CXM_ASSERT(out_h <= 0 || out_w <= 0, "Convolution output size became non-positive. Check kernel/stride/padding.");

    return {out_h, out_w};
}

tensor Conv2D::im2col(const tensor& input) const {
    const i64 N  = input.shape()[0];
    const i64 C  = input.shape()[1];
    const i64 H  = input.shape()[2];
    const i64 W  = input.shape()[3];

    const i64 oH = (H + 2 * this->PADDING_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    const i64 oW = (W + 2 * this->PADDING_WIDTH  - this->KERNEL_WIDTH)  / this->STRIDE_WIDTH  + 1;

    const i64 rows = C * this->KERNEL_HEIGHT * this->KERNEL_WIDTH;
    const i64 cols = N * oH * oW;

    Tensor output({rows, cols}, input.device(), input.has_grad());

    ix::Convolution::unfold(
        input.get(), output.get(),
        N, C, H, W,
        this->KERNEL_HEIGHT, this->KERNEL_WIDTH,
        this->STRIDE_HEIGHT, this->STRIDE_WIDTH,
        this->PADDING_HEIGHT, this->PADDING_WIDTH,
        oH, oW,
        input.device()
    );

    if (output.has_grad()) [[likely]] {
        output.SetFlow(std::make_shared<meta::conv2d>(input.pack(), H, W, this->KERNEL_HEIGHT, this->KERNEL_WIDTH, this->STRIDE_HEIGHT, this->STRIDE_WIDTH, this->PADDING_HEIGHT, this->PADDING_WIDTH, oH, oW));
    }

    return output;
}