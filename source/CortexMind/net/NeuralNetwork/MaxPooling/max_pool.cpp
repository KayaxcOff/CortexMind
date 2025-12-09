//
// Created by muham on 7.12.2025.
//

#include "CortexMind/net/NeuralNetwork/MaxPooling/max_pool.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

MaxPooling::MaxPooling(const int kernel_width, const int kernel_height, const int stride_width, const int stride_height, std::unique_ptr<ActivationFunc> activation_func) : Layer(std::move(activation_func)), KERNEL_WIDTH(kernel_width), KERNEL_HEIGHT(kernel_height), STRIDE_WIDTH(stride_width), STRIDE_HEIGHT(stride_height), idx(), INPUT_WIDTH(0), INPUT_HEIGHT(0), INPUT_CHANNELS(0) {}

MaxPooling::~MaxPooling() = default;

tensor MaxPooling::forward(const tensor &input) {
    this->input_cache = input;

    const int batch = input.shapeIdx(0);
    this->INPUT_CHANNELS = input.shapeIdx(1);
    this->INPUT_HEIGHT = input.shapeIdx(2);
    this->INPUT_WIDTH = input.shapeIdx(3);

    const size_t out_h = (this->INPUT_HEIGHT - this->KERNEL_HEIGHT) / this->STRIDE_HEIGHT + 1;
    const size_t out_w = (this->INPUT_WIDTH - this->KERNEL_WIDTH) / this->STRIDE_WIDTH + 1;

    tensor output(batch, this->INPUT_CHANNELS, static_cast<int>(out_h), static_cast<int>(out_w));
    this->idx = tensor(batch, this->INPUT_CHANNELS, static_cast<int>(out_h), static_cast<int>(out_w));

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->INPUT_CHANNELS; ++j) {
            for (int k = 0; k < out_h; ++k) {
                for (int l = 0; l < out_w; ++l) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t maxIdx = 0;

                    for (int m = 0; m < this->KERNEL_HEIGHT; ++m) {
                        for (int n = 0; n < this->KERNEL_WIDTH; ++n) {
                            const int in_row = k * this->STRIDE_HEIGHT + m;
                            const int in_col = l * this->STRIDE_WIDTH + n;
                            const double val = input.at(i, j, in_row, in_col);
                            const size_t currIdx = m * this->KERNEL_WIDTH + n;

                            if (val > max_val) {
                                max_val = val;
                                maxIdx = currIdx;
                            }
                        }
                    }
                    output.at(i, j, k, l) = static_cast<float>(max_val);
                    this->idx.at(i, j, k, l) = static_cast<float>(maxIdx);
                }
            }
        }
    }
    return output;
}

tensor MaxPooling::backward(const tensor &grad_output) {
    auto grad_input = tensor(grad_output.shapeIdx(0), this->INPUT_CHANNELS, this->INPUT_HEIGHT, this->INPUT_WIDTH);

    const int batch = grad_output.shapeIdx(0);
    const size_t out_h = grad_output.shapeIdx(2);
    const size_t out_w = grad_output.shapeIdx(3);

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->INPUT_CHANNELS; ++j) {
            for (int k = 0; k < out_h; ++k) {
                for (int l = 0; l < out_w; ++l) {
                    const float sum = grad_output.at(i, j, k, l);

                    const auto maxIdx = static_cast<size_t>(this->idx.at(i, j, k, l));

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

std::string MaxPooling::config() const {
    return "MaxPooling";
}

std::vector<std::reference_wrapper<tensor> > MaxPooling::gradients() {
    return {};
}

std::vector<std::reference_wrapper<tensor> > MaxPooling::parameters() {
    return {};
}
