//
// Created by muham on 12.12.2025.
//

#include "CortexMind/net/NeuralNetwork/MaxPooling/max_pool.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

MaxPooling::MaxPooling(const int kernel_width, const int kernel_height, const int stride_width, const int stride_height) : KERNEL_WIDTH(kernel_width), KERNEL_HEIGHT(kernel_height), STRIDE_WIDTH(stride_width), STRIDE_HEIGHT(stride_height), INPUT_CHANNELS(0), INPUT_WIDTH(0), INPUT_HEIGHT(0) {}

tensor MaxPooling::forward(const tensor &input) {
    this->input_cache = input;

    const int batch = input.batch();
    this->INPUT_CHANNELS = input.channel();
    this->INPUT_WIDTH = input.width();
    this->INPUT_HEIGHT = input.height();

    const size_t out_h = (this->INPUT_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    const size_t out_w = (this->INPUT_WIDTH - this->KERNEL_WIDTH) / this->STRIDE_WIDTH + 1;

    tensor output(batch, this->INPUT_CHANNELS, static_cast<int>(out_w), static_cast<int>(out_h));
    this->idx.resize(batch, this->INPUT_CHANNELS, static_cast<int>(out_w), static_cast<int>(out_h));

    output.zero();
    this->idx.zero();

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->INPUT_CHANNELS; ++j) {
            for (int k = 0; k < out_h; ++k) {
                for (int l = 0; l < out_w; ++l) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t maxIdx = 0;

                    for (int m = 0; m < this->KERNEL_HEIGHT; ++m) {
                        for (int n = 0; n < this->KERNEL_WIDTH; ++n) {
                            const int in_y = k * this->STRIDE_HEIGHT + m;
                            const int in_x = l * this->STRIDE_WIDTH + n;

                            const double val = input.at(i, j, in_y, in_x);
                            if (val > max_val) {
                                max_val = val;
                                maxIdx = m * this->KERNEL_WIDTH + n;
                            }
                        }
                    }
                    output.at(i, j, l, k) = static_cast<float>(max_val);
                    this->idx.at(i, j, l, k) = static_cast<float>(maxIdx);
                }
            }
        }
    }
    return output;
}

tensor MaxPooling::backward(const tensor &grad_output) {
    tensor grad_input(grad_output.batch(), this->INPUT_CHANNELS, this->INPUT_WIDTH, this->INPUT_HEIGHT);

    const int batch = grad_output.batch();
    const size_t out_h = grad_output.height();
    const size_t out_w = grad_output.width();

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->INPUT_CHANNELS; ++j) {
            for (int k = 0; k < out_h; ++k) {
                for (int l = 0; l < out_w; ++l) {
                    const float sum = grad_output.at(i, j, l, k);

                    const auto maxIdx = static_cast<size_t>(this->idx.at(i, j, l, k));

                    const int m = static_cast<int>(maxIdx / this->KERNEL_WIDTH);
                    const int n = static_cast<int>(maxIdx % this->KERNEL_WIDTH);

                    const int in_row = k * this->STRIDE_HEIGHT + m;
                    const int in_col = l * this->STRIDE_WIDTH + n;

                    grad_input.at(i, j, in_row, in_col) += sum;
                }
            }
        }
    }
    return grad_input;
}

std::string MaxPooling::config() {
    return "MaxPooling";
}
