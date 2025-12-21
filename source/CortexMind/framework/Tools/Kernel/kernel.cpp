//
// Created by muham on 10.12.2025.
//

#include "CortexMind/framework/Tools/Kernel/kernel.hpp"
#include <CortexMind/framework/Core/AVX/matrix.hpp>
#include <cmath>

using namespace cortex::_fw;
using namespace cortex;

MindKernel::MindKernel(const int in_channel, const int out_channel, const int kernel_height, const int kernel_width) : IN_CHANNEL(in_channel), OUT_CHANNEL(out_channel), KERNEL_HEIGHT(kernel_height), KERNEL_WIDTH(kernel_width) {
    this->weights.allocate(out_channel, in_channel, kernel_height, kernel_width);
    this->grad_weights.allocate(out_channel, in_channel, kernel_height, kernel_width);

    const float limit = std::sqrt(2.0f / static_cast<float>(in_channel * kernel_height * kernel_width));
    this->weights.uniform_rand(-limit, limit);
    this->grad_weights.zero();
}

MindKernel::~MindKernel() = default;

tensor MindKernel::apply(const tensor &input) {
    const int batch = input.batch();
    const int height = input.height();
    const int width = input.width();

    const int out_height = height - this->KERNEL_HEIGHT + 1;
    const int out_width = width - this->KERNEL_WIDTH + 1;

    tensor output(batch, this->OUT_CHANNEL, out_height, out_width);

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->OUT_CHANNEL; ++j) {
            for (int k = 0; k < out_height; ++k) {
                for (int l = 0; l < out_width; ++l) {
                    avx2::reg acc = avx2::zero();

                    for (int m = 0; m < this->IN_CHANNEL; ++m) {
                        for (int n = 0; n < this->KERNEL_HEIGHT; ++n) {
                            const int sum = k + n;

                            int kw = 0;
                            while (kw + 8 < this->KERNEL_WIDTH) {
                                const size_t inIdx = ((i * this->IN_CHANNEL + m) * height + sum) * width + (l + kw);
                                const size_t wIdx = ((j * this->IN_CHANNEL + m) * this->KERNEL_HEIGHT + n) * this->KERNEL_WIDTH + kw;

                                const avx2::reg v_in = avx2::load(input.raw_ptr(inIdx));
                                const avx2::reg v_w = avx2::load(this->weights.raw_ptr(wIdx));

                                acc = avx2::fma(v_in, v_w, acc);
                                kw += 8;
                            }

                            while (kw < this->KERNEL_WIDTH) {
                                const float in_val = input.at(i, m, sum, l + kw);
                                const float w_val = this->weights.at(j, m, n, kw);
                                output.at(i, j, k, l) += in_val * w_val;
                                kw++;
                            }
                        }
                    }
                    output.at(i, j, k, l) += avx2::h_sum(acc);
                }
            }
        }
    }
    return output;
}

tensor MindKernel::backward(const tensor &input, const tensor &grad_output) {
    const int batch = input.batch();
    const int height = input.height();
    const int width = input.width();

    const int out_height = height - this->KERNEL_HEIGHT + 1;
    const int out_width = width - this->KERNEL_WIDTH + 1;

    tensor grad_input(batch, this->IN_CHANNEL, height, width, 0.0f);

    for (int i = 0; i < this->OUT_CHANNEL; ++i) {
        for (int j = 0; j < this->IN_CHANNEL; ++j) {
            for (int k = 0; k < this->KERNEL_HEIGHT; ++k) {
                for (int l = 0; l < this->KERNEL_WIDTH; ++l) {
                    avx2::reg acc = avx2::zero();

                    for (int m = 0; m < batch; ++m) {
                        for (int n = 0; n < out_height; ++n) {
                            const int sum = k + n;

                            int ow = 0;
                            while (ow + 8 <= out_width) {
                                alignas(32) float in_vals[8];
                                alignas(32) float grad_vals[8];

                                for (int t = 0; t < 8; ++t) {
                                    in_vals[t] = input.at(m, j, n, ow + l + t);
                                    grad_vals[t] = grad_output.at(m, i, n, ow + t);
                                }

                                const avx2::reg v_in = avx2::load(in_vals);
                                const avx2::reg v_grad = avx2::load(grad_vals);

                                acc = avx2::fma(v_in, v_grad, acc);
                                ow += 8;
                            }

                            while (ow < out_width) {
                                const float in_val = input.at(m, j, j, ow + l);
                                const float grad_val = grad_output.at(m, i, sum, ow);
                                this->grad_weights.at(i, j, k, l) += in_val * grad_val;
                                ow++;
                            }
                        }
                    }
                    this->grad_weights.at(i, j, k, l) += avx2::h_sum(acc);
                }
            }
        }
    }

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->IN_CHANNEL; ++j) {
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    float sum = 0.0f;

                    for (int m = 0; m < this->OUT_CHANNEL; ++m) {
                        for (int n = 0; n < this->KERNEL_HEIGHT; ++n) {
                            const int oh = k - n;
                            if (oh < 0 || oh >= out_height) continue;

                            for (int kw = 0; kw < this->KERNEL_WIDTH; ++kw) {
                                const int ow = l - kw;
                                if (ow < 0 || ow >= out_width) continue;

                                sum += this->weights.at(m, j, n, kw) * grad_output.at(i, m, oh, ow);
                            }
                        }
                    }
                    grad_input.at(i, j, k, l) = sum;
                }
            }
        }
    }
    return grad_input;
}

void MindKernel::zero_grad() noexcept {
    this->grad_weights.zero();
}

std::array<tensor*, 1> MindKernel::parameters() noexcept {
    return {&this->weights};
}

std::array<tensor*, 1> MindKernel::gradients() noexcept {
    return {&this->grad_weights};
}