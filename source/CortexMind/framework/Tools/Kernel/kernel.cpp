//
// Created by muham on 10.12.2025.
//

#include "CortexMind/framework/Tools/Kernel/kernel.hpp"
#include <CortexMind/framework/Core/AVX/matrix.hpp>

using namespace cortex::_fw;
using namespace cortex;

MindKernel::MindKernel(const int in_channel, const int out_channel, const int kernel_height, const int kernel_width, const float value) : weights(out_channel, in_channel, kernel_height, kernel_width, value), IN_CHANNEL(in_channel), OUT_CHANNEL(out_channel), KERNEL_HEIGHT(kernel_height), KERNEL_WIDTH(kernel_width) {
    this->weights.uniform_rand();
}

MindKernel::~MindKernel() = default;

tensor MindKernel::apply(const tensor &input) {
    const int batch = input.batch();
    const int height = input.height();
    const int width = input.width();

    const int out_height = height - this->KERNEL_HEIGHT + 1;
    const int out_width = width - this->KERNEL_WIDTH + 1;

    tensor output(batch, this->OUT_CHANNEL, out_height, out_width, 0.0f);
    alignas(32) float tmp[8];

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < this->OUT_CHANNEL; ++j) {
            for (int k = 0; k < out_height; ++k) {
                for (int l = 0; l < out_width; ++l) {
                    float sum = 0.0f;

                    for (int m = 0; m < this->IN_CHANNEL; ++m) {
                        for (int n = 0; n < this->KERNEL_HEIGHT; ++n) {
                            const int ih = k + n;

                            int flag = 0;
                            while (flag < this->KERNEL_WIDTH) {
                                const int rem = std::min(8, this->KERNEL_WIDTH - flag);

                                const float* in_ptr = &input.at(i, m, ih, l + flag);
                                const float* w_ptr = &this->weights.at(j, m, n, flag);

                                avx2::matrix_t::mul(in_ptr, w_ptr, tmp, rem);

                                for (int t = 0; t < rem; ++t) sum += tmp[t];
                                flag += rem;
                            }
                        }
                    }
                    output.at(i, j, k, l) = sum;
                }
            }
        }
    }
    return output;
}

tensor MindKernel::backward(const tensor &in, const tensor &grad_out) {
    const int batch = in.batch();
    const int channel = in.channel();
    const int height = in.height();
    const int width = in.width();

    const int out_height = height - this->KERNEL_HEIGHT + 1;
    const int out_width = width - this->KERNEL_WIDTH + 1;

    tensor grad_in(batch, channel, height, width, 0.0f);
    tensor grad_weight(batch, channel, out_height, out_width, 0.0f);

    alignas(32) float tmp_a[8];
    alignas(32) float tmp_b[8];
    alignas(32) float tmp_res[8];

    for (int i = 0; i < this->OUT_CHANNEL; ++i) {
        for (int j = 0; j < this->IN_CHANNEL; ++j) {
            for (int k = 0; k < this->KERNEL_HEIGHT; ++k) {
                for (int l = 0; l < this->KERNEL_WIDTH; ++l) {
                    float sum = 0.0f;

                    for (int m = 0; m < batch; ++m) {
                        for (int n = 0; n < out_height; ++n) {
                            const int ih = n + k;

                            int flag = 0;
                            while (flag < out_width) {
                                const int rem = std::min(8, out_width - flag);

                                for (int t = 0; t < rem; ++t) {
                                    tmp_a[t] = in.at(i, j, ih, flag + t);
                                    tmp_b[t] = grad_out.at(i, i, n, flag + t);
                                }

                                avx2::matrix_t::mul(tmp_a, tmp_b, tmp_res, rem);

                                for (int t = 0; t < rem; ++t) sum += tmp_res[t];

                                flag += rem;
                            }
                        }
                    }
                    grad_weight.at(i, j, k, l) = sum;
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
                            const int ih = k - n;
                            if (ih < 0 || ih >= out_height) continue;

                            int flag = 0;
                            while (flag < this->KERNEL_WIDTH) {
                                const int rem = std::min(8, this->KERNEL_WIDTH - flag);
                                for (int t = 0; t < rem; ++t) {
                                    tmp_a[t] = this->weights.at(m, j, n, flag + t);
                                    tmp_b[t] = grad_out.at(i, m, ih, l - flag + t);
                                }

                                avx2::matrix_t::mul(tmp_a, tmp_b, tmp_res, rem);

                                for (int t = 0; t < rem; ++t) sum += tmp_res[t];

                                flag += rem;
                            }
                        }
                    }
                    grad_in.at(i, j, k, l) = sum;
                }
            }
        }
    }
    return grad_in;
}
