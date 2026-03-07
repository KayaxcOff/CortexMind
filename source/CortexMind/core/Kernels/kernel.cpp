//
// Created by muham on 26.02.2026.
//

#include "CortexMind/core/Kernels/kernel.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::_fw;
using namespace cortex;

Kernel::Kernel(const i64 in_channel, const i64 out_channel, const i64 kernel_width, const i64 kernel_height, const i64 stride, const i64 padding, const bool _requires_grad, const sys::device _dev) : INPUT_CHANNEL(in_channel), OUTPUT_CHANNEL(out_channel), KERNEL_WIDTH(kernel_width), KERNEL_HEIGHT(kernel_height), m_padding(padding), m_stride(stride){
    this->weight = tensor({this->OUTPUT_CHANNEL, this->INPUT_CHANNEL * this->KERNEL_WIDTH * this->KERNEL_HEIGHT}, _dev, _requires_grad);
    this->weight.uniform_rand();

    this->bias = tensor({this->OUTPUT_CHANNEL, 1}, _dev, _requires_grad);
    this->bias.zero();
}

tensor Kernel::forward(tensor &in) {
    CXM_ASSERT(in.ndim() == 4, "cortex::_fw::Kernel::forward()", "Input must be 4D: {batch, in_ch, H, W}.");
    CXM_ASSERT(in.shape()[1] == this->INPUT_CHANNEL, "cortex::_fw::Kernel::forward()", "Input channel mismatch.");

    const i64 batch = in.shape()[0];
    const i64 H     = in.shape()[2];
    const i64 W     = in.shape()[3];
    const i64 H_out = this->getOutputHeight(H);
    const i64 W_out = this->getOutputWidth(W);

    CXM_ASSERT(H_out > 0 && W_out > 0, "cortex::_fw::Kernel::forward()", "Output dimensions must be positive. Check kernel size, stride, padding.");

    tensor output({batch, this->OUTPUT_CHANNEL, H_out, W_out}, in.devices(), in.requires_grad() || this->weight.requires_grad());

    for (i64 i = 0; i < batch; ++i) {
        tensor input_b({this->INPUT_CHANNEL, H, W}, in.devices(), false);
        for (i64 c = 0; c < this->INPUT_CHANNEL; ++c) {
            for (i64 h = 0; h < H; ++h) {
                for (i64 w = 0; w < W; ++w) {
                    input_b.at(c, h, w) = in.at(i, c, h, w);
                }
            }
        }
        tensor col = this->im2col(input_b, H_out, W_out);
        tensor out_b = this->weight.matmul(col);

        tensor bias_expanded = this->bias.repeat(H_out * W_out, 1);
        out_b = out_b + bias_expanded;

        for (i64 oc = 0; oc < this->OUTPUT_CHANNEL; ++oc) {
            for (i64 oh = 0; oh < H_out; ++oh) {
                for (i64 ow = 0; ow < W_out; ++ow) {
                    output.at(i, oc, oh, ow) = out_b.at(oc, oh * W_out + ow);
                }
            }
        }
    }
    return output;
}

tensor &Kernel::getWeight() {
    return this->weight;
}

tensor &Kernel::getBias() {
    return this->bias;
}

i64 Kernel::getOutputHeight(const i64 input_h) const noexcept {
    return (input_h + 2 * this->m_padding - this->KERNEL_HEIGHT) / this->m_stride + 1;
}

i64 Kernel::getOutputWidth(const i64 input_w) const noexcept {
    return (input_w + 2 * this->m_padding - this->KERNEL_WIDTH) / this->m_stride + 1;
}

tensor Kernel::im2col(const tensor &input, const i64 H_out, const i64 W_out) const {
    const i64 height = input.shape()[1];
    const i64 width = input.shape()[2];
    i64 col_rows = this->INPUT_CHANNEL * this->KERNEL_WIDTH * this->KERNEL_HEIGHT;
    i64 col_cols = H_out * W_out;

    tensor output({col_rows, col_cols}, input.devices(), input.requires_grad());

    for (i64 i = 0; i < H_out; ++i) {
        for (i64 j = 0; j < W_out; ++j) {
            const i64 colIdx = i * W_out + j;
            for (i64 k = 0; k < this->INPUT_CHANNEL; ++k) {
                for (i64 l = 0; l < this->KERNEL_HEIGHT; ++l) {
                    for (i64 m = 0; m < this->KERNEL_WIDTH; ++m) {
                        const i64 row = k * this->KERNEL_HEIGHT * this->KERNEL_WIDTH + l * this->KERNEL_WIDTH + m;
                        const i64 ih = i * this->m_stride - this->m_padding + l;
                        const i64 iw = j * this->m_stride - this->m_padding + m;

                        output.at(row, colIdx) = (ih >= 0 && ih < height && iw >= 0 && iw < width) ? input.at(k, ih, iw) : 0;
                    }
                }
            }
        }
    }
    return output;
}
