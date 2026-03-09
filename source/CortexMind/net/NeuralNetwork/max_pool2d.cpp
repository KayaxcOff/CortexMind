//
// Created by muham on 9.03.2026.
//

#include "CortexMind/net/NeuralNetwork/max_pool2d.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <limits>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

MaxPooling2D::MaxPooling2D(const int64 kernel_size, const int64 stride) : Layer(true, "MaxPooling2D"), KERNEL_SIZE(kernel_size), STRIDE(stride == -1 ? kernel_size : stride) {}

MaxPooling2D::~MaxPooling2D() = default;

tensor MaxPooling2D::forward(tensor &input) {
    CXM_ASSERT(input.ndim() == 4, "cortex::nn::MaxPooling2D::forward()", "Input must be 4D: {batch, channels, H, W}.");

    this->last_input = input;
    this->last_input.clear_flow();

    const int64 batch = input.shape()[0];
    const int64 C     = input.shape()[1];
    const int64 H     = input.shape()[2];
    const int64 W     = input.shape()[3];
    const int64 H_out = (H - this->KERNEL_SIZE) / this->STRIDE + 1;
    const int64 W_out = (W - this->KERNEL_SIZE) / this->STRIDE + 1;

    CXM_ASSERT(H_out > 0 && W_out > 0, "cortex::nn::MaxPooling2D::forward()", "Output dimensions must be positive.");

    tensor output({batch, C, H_out, W_out}, input.devices(), input.requires_grad());

    const int64 total = batch * C * H_out * W_out;
    this->maxIndices.assign(total, 0);

    for (int64 b = 0; b < batch; ++b) {
        for (int64 c = 0; c < C; ++c) {
            for (int64 oh = 0; oh < H_out; ++oh) {
                for (int64 ow = 0; ow < W_out; ++ow) {
                    float32 max_val = std::numeric_limits<float32>::lowest();
                    int64 max_h = 0, max_w = 0;

                    for (int64 kh = 0; kh < this->KERNEL_SIZE; ++kh) {
                        for (int64 kw = 0; kw < this->KERNEL_SIZE; ++kw) {
                            const int64 ih = oh * this->STRIDE + kh;
                            const int64 iw = ow * this->STRIDE + kw;
                            const float32 val = input.at(b, c, ih, iw);
                            if (val > max_val) {
                                max_val = val;
                                max_h   = ih;
                                max_w   = iw;
                            }
                        }
                    }

                    output.at(b, c, oh, ow) = max_val;

                    const int64 idx = ((b * C + c) * H_out + oh) * W_out + ow;
                    this->maxIndices[idx] = max_h * W + max_w;
                }
            }
        }
    }
    return output;
}

std::vector<ref<tensor>> MaxPooling2D::parameters() {
    return {};
}

std::vector<ref<tensor> > MaxPooling2D::gradients() {
    return {};
}
