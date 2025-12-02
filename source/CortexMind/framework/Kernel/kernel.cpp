//
// Created by muham on 30.11.2025.
//

#include "CortexMind/framework/Kernel/kernel.hpp"
#include <cmath>

using namespace cortex::fw;
using namespace cortex;

MindKernel::MindKernel(size_t c_in, const size_t c_out, const size_t kernel_size, const size_t _stride, const size_t _padding, const bool required_grad) : weights(0,0,0), bias(0,0,0), gradWeights(0,0,0), gradBias(0,0,0), stride(_stride), padding(_padding), K(kernel_size), C_in(1), C_out(c_out), padded_input(0,0,0) {
    this->weights = tensor(c_out, this->K, this->K, required_grad);
    this->bias = tensor(1, 1, c_out, required_grad);

    this->gradWeights = tensor(c_out, this->K, this->K, required_grad);
    this->gradBias = tensor(1, 1, c_out, required_grad);

    const double limit = std::sqrt(1.0 / static_cast<double>(this->K * this->K));
    this->weights.uniform_rand(-limit, limit);
    this->bias.zero();
}

MindKernel::~MindKernel() = default;

tensor MindKernel::apply(const tensor &input) {
    const size_t B = input.get_shape()[0];
    this->H_in = input.get_shape()[1];
    this->W_in = input.get_shape()[2];

    const size_t H_pad = this->H_in + 2 * this->padding;
    const size_t W_pad = this->W_in + 2 * this->padding;

    this->padded_input = tensor(B, H_pad, W_pad, false);
    this->padded_input.zero();

    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < this->H_in; j++) {
            for (size_t k = 0; k < this->W_in; k++) {
                this->padded_input(i, j + this->padding, k + this->padding) = input(i, j, k);
            }
        }
    }

    const size_t H_out = (H_pad - this->K) / this->stride + 1;
    const size_t W_out = (W_pad - this->K) / this->stride + 1;

    tensor output(B, H_out, W_out * this->C_out, false);

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < this->C_out; ++j) {
            for (size_t k = 0; k < H_out; ++k) {
                for (size_t l = 0; l < W_out; ++l) {
                    double sum = this->bias(0, 0, j);

                    const size_t h_start = k * this->stride;
                    const size_t w_start = l * this->stride;

                    for (size_t kh = 0; kh < this->K; ++kh) {
                        for (size_t kw = 0; kw < this->K; ++kw) {
                            sum += this->padded_input(i, h_start + kh, w_start + kw)
                                   * this->weights(j, kh, kw);
                        }
                    }

                    const size_t col_idx = l * this->C_out + j;
                    output(i, k, col_idx) = sum;
                }
            }
        }
    }
    return output;
}

tensor MindKernel::backward(const tensor& input, const tensor& grad_output) {
    const size_t B = input.get_shape()[0];
    const size_t H_out = grad_output.get_shape()[1];
    const size_t W_out = grad_output.get_shape()[2] / this->C_out;

    tensor grad_input(B, this->H_in, this->W_in, false);
    grad_input.zero();

    this->gradWeights.zero();
    this->gradBias.zero();

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < this->C_out; ++j) {
            for (size_t k = 0; k < H_out; ++k) {
                for (size_t n = 0; n < W_out; ++n) {
                    const double dy = grad_output(i, k, n * this->C_out + j);

                    this->gradBias(0,0,j) += dy;

                    const size_t h_start = k * this->stride;
                    const size_t w_start = n * this->stride;

                    for (size_t l = 0; l < this->K; ++l) {
                        for (size_t w = 0; w < this->K; ++w) {
                            this->gradWeights(j, l, w) += this->padded_input(i, h_start + l, w_start + w) * dy;

                            const size_t ih = h_start + l;

                            if (const size_t iw = w_start + w; ih >= this->padding && ih < this->H_in + this->padding && iw >= this->padding && iw < this->W_in + this->padding) {
                                grad_input(i, ih - this->padding, iw - this->padding) += this->weights(j, l, w) * dy;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}
