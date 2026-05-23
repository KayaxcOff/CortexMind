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

namespace {
    void im2col_cpu(
    const f32* input,         // (N, C, H, W)
    f32*       output,        // (C*kH*kW, N*oH*oW)
    i64 N,  i64 C,  i64 H,  i64 W,
    i64 kH, i64 kW,
    i64 sH, i64 sW,
    i64 pH, i64 pW,
    i64 oH, i64 oW)
    {
        // output layout: satır = (c, kh, kw), sütun = (n, oh, ow)
        for (i64 c = 0; c < C; ++c) {
            for (i64 kh = 0; kh < kH; ++kh) {
                for (i64 kw = 0; kw < kW; ++kw) {
                    const i64 row = c * kH * kW + kh * kW + kw;

                    for (i64 n = 0; n < N; ++n) {
                        for (i64 oh = 0; oh < oH; ++oh) {
                            for (i64 ow = 0; ow < oW; ++ow) {
                                const i64 col = n * oH * oW + oh * oW + ow;

                                const i64 ih = oh * sH - pH + kh;
                                const i64 iw = ow * sW - pW + kw;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    output[row * (N*oH*oW) + col] =
                                        input[n*(C*H*W) + c*(H*W) + ih*W + iw];
                                } else {
                                    output[row * (N*oH*oW) + col] = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
} //unnamed namespace

Conv2D::Conv2D(int64 in_channels, int64 out_channels, int64 kH, int64 kW, const int64 sH, const int64 sW, const int64 pH, const int64 pW, const sys::DeviceType device) : LayerBase("Conv2D(" + std::to_string(out_channels) + ")"), KERNEL_WIDTH(kW), KERNEL_HEIGHT(kH) , STRIDE_WIDTH(sW), STRIDE_HEIGHT(sH) , PADDING_WIDTH(pW), PADDING_HEIGHT(pH) {
    this->weight = Tensor({out_channels, in_channels, kH, kW}, device, true);
    this->bias   = Tensor({out_channels}, device, true);

    const float64 limit = std::sqrt(6.0 / (static_cast<float64>(in_channels * kH * kW)));
    this->weight.uniform(-static_cast<float32>(limit), static_cast<float32>(limit));
    this->bias.zero();
}

Conv2D::~Conv2D() = default;

tensor Conv2D::forward(const tensor &input) {
    CXM_ASSERT(input.ndim() != 4,
        "Conv2D expects 4D input (N,C,H,W)");
    CXM_ASSERT(input.shape()[1] != this->weight.shape()[1],
        "Input channels mismatch");

    const i64 N  = input.shape()[0];
    const i64 H  = input.shape()[2];
    const i64 W  = input.shape()[3];
    const i64 oC = this->weight.shape()[0];

    auto [oH, oW] = compute_output_size(H, W);

    // im2col: (C*kH*kW, N*oH*oW)
    const Tensor col = this->im2col(input);

    // weight_flat: (oC, C*kH*kW)
    const Tensor weight_flat = this->weight.reshape({oC, -1});

    // matmul: (oC, C*kH*kW) @ (C*kH*kW, N*oH*oW) = (oC, N*oH*oW)
    Tensor output = weight_flat.matmul(col);

    // bias broadcast: (oC, 1) + (oC, N*oH*oW) → row broadcast
    output = output + this->bias.reshape({oC, 1});

    // reshape: (oC, N*oH*oW) → (N, oC, oH, oW)
    // önce (N, oC, oH, oW) için permute gerekiyor
    // matmul çıktısı (oC, N*oH*oW) — bunu (N, oC, oH, oW)'e çevirmek
    // reshape({oC, N, oH, oW}) → permute({1,0,2,3}) → (N, oC, oH, oW)
    output = output.reshape({oC, N, oH, oW}).permute({1, 0, 2, 3});

    // contiguous kopyası gerekiyor — permute non-contiguous
    // şimdilik clone ile çözelim
    return output.clone();
}

std::vector<ref<tensor>> Conv2D::getParameters() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Conv2D::getGradients() {
    return {this->weight.grad(), this->bias.grad()};
}

tensor Conv2D::im2col(const tensor& input) const {
    const i64 N = input.shape()[0];
    const i64 C = input.shape()[1];
    const i64 H = input.shape()[2];
    const i64 W = input.shape()[3];

    const i64 oH = (H + 2 * this->PADDING_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    const i64 oW = (W + 2 * this->PADDING_WIDTH  - this->KERNEL_WIDTH)  / this->STRIDE_WIDTH  + 1;

    const i64 rows = C * this->KERNEL_HEIGHT * this->KERNEL_WIDTH;
    const i64 cols = N * oH * oW;

    Tensor output({rows, cols}, input.device(), false);

    #if CXM_IS_CUDA_AVAILABLE
        if (input.device() == sys::DeviceType::kCUDA) {
            // CUDA kernel — sonra eklenecek
            CXM_ASSERT(true, "im2col CUDA not implemented yet");
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE

    im2col_cpu(input.get(), output.get(),
               N, C, H, W,
               this->KERNEL_HEIGHT, this->KERNEL_WIDTH,
               this->STRIDE_HEIGHT, this->STRIDE_WIDTH,
               this->PADDING_HEIGHT, this->PADDING_WIDTH,
               oH, oW);

    return output;
}

std::pair<int64, int64> Conv2D::compute_output_size(const int64 input_height, const int64 input_width) const {
    int64 out_h = (input_height + 2 * this->PADDING_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    int64 out_w = (input_width  + 2 * this->PADDING_WIDTH  - this->KERNEL_WIDTH)  / this->STRIDE_WIDTH  + 1;

    CXM_ASSERT(out_h <= 0 || out_w <= 0, "Convolution output size became non-positive. Check kernel/stride/padding.");

    return {out_h, out_w};
}