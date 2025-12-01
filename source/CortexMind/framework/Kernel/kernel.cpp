//
// Created by muham on 30.11.2025.
//

#include "CortexMind/framework/Kernel/kernel.hpp"
#include <cmath>

using namespace cortex::tools;
using namespace cortex;

MindKernel::MindKernel(const size_t c_in, const size_t c_out, const size_t kernel_size, const size_t _stride, const size_t _padding, const bool required_grad) : weights(0, 0, 0), bias(0, 0, 0), gradWeights(0, 0, 0), gradBias(0, 0, 0), stride(_stride), padding(_padding), K(kernel_size), C_in(c_in), C_out(c_out), padded_input(0, 0, 0) {
    this->weights = tensor(c_out, this->K, this->K, required_grad);
    this->bias = tensor(1, 1, c_out, required_grad);

    this->gradWeights = tensor(c_out, this->K, this->K, required_grad);
    this->gradBias = tensor(1, 1, c_out, false);

    const double limit = std::sqrt(2.0 / static_cast<double>(this->K * this->K * this->C_in));
    this->weights.uniform_rand(-limit, limit);
    this->bias.fill(0.0);
}

MindKernel::~MindKernel() = default;

tensor MindKernel::apply(const tensor &input) {
    const size_t B = input._shape()[0];
    this->H_in = input._shape()[1];
    this->W_in = input._shape()[2];

    const size_t H_out = (this->H_in - this->K + 2 * this->padding) / this->stride + 1;
    const size_t W_out = (this->W_in - this->K + 2 * this->padding) / this->stride + 1;

    tensor output(B, H_out, W_out * this->C_out, false);

    const tensor& paddedInput = input;

    for (size_t b = 0; b < B; ++b) {
        for (size_t c_out = 0; c_out < this->C_out; ++c_out) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    double sum = 0.0;

                    const size_t h_start = h_out * this->stride;
                    const size_t w_start = w_out * this->stride;

                    for (size_t k_h = 0; k_h < this->K; ++k_h) {
                        for (size_t k_w = 0; k_w < this->K; ++k_w) {
                            const size_t input_h = h_start + k_h;
                            const size_t input_w = w_start + k_w;

                            sum += paddedInput(b, input_h, input_w * this->C_in + 0) * this->weights(c_out, k_h, k_w);
                        }
                    }
                    sum += this->bias(0, 0, c_out);
                    const size_t output_col_idx = w_out * C_out + c_out;
                    output(b, h_out, output_col_idx) = sum;
                }
            }
        }
    }
    this->padded_input = paddedInput;
    return output;
}

tensor MindKernel::backward(const tensor &input, const tensor &grad_output) {
    const size_t B = input._shape()[0];
    const size_t H_out = grad_output._shape()[1];
    const size_t W_out = grad_output._shape()[2] / C_out;

    tensor grad_input(B, this->H_in, this->W_in, false);

    this->gradWeights.zero();
    this->gradBias.zero();

    for (size_t b = 0; b < B; ++b) {
        for (size_t c_out = 0; c_out < this->C_out; ++c_out) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    const size_t idx = w_out * C_out + c_out;
                    this->gradBias(0, 0, c_out) += grad_output(b, h_out, idx);
                }
            }

            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    const size_t h_start = h_out * this->stride;
                    const size_t w_start = w_out * this->stride;
                    const size_t idx = w_out * C_out + c_out;

                    const double dy = grad_output(b, h_out, idx);

                    for (size_t k_h = 0; k_h < this->K; ++k_h) {
                        for (size_t k_w = 0; k_w < this->K; ++k_w) {
                            const size_t input_h = h_start + k_h;
                            const size_t input_w = w_start + k_w;

                            this->gradWeights(c_out, k_h, k_w) += this->padded_input(b, input_h, input_w) * dy;
                            grad_input(b, input_h, input_w) += this->weights(c_out, k_h, k_w) * dy;
                        }
                    }
                }
            }
        }
    }
    return grad_input;
}
