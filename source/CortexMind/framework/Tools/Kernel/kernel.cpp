//
// Created by muham on 10.12.2025.
//

#include "CortexMind/framework/Tools/Kernel/kernel.hpp"
#include <CortexMind/framework/Core/AVX/matrix.hpp>

using namespace cortex::_fw;
using namespace cortex;

MindKernel::MindKernel(const int in_channel, const int out_channel, const int kernel_height, const int kernel_width) : weights(out_channel, in_channel, kernel_height, kernel_width), IN_CHANNEL(in_channel), OUT_CHANNEL(out_channel), KERNEL_HEIGHT(kernel_height), KERNEL_WIDTH(kernel_width) {
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

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < this->OUT_CHANNEL; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;

                    for (int ic = 0; ic < this->IN_CHANNEL; ++ic) {
                        for (int kh = 0; kh < this->KERNEL_HEIGHT; ++kh) {
                            const int ih = oh + kh;

                            int kw = 0;
                            while (kw < this->KERNEL_WIDTH) {
                                const int rem = std::min(8, this->KERNEL_WIDTH - kw);

                                const float* in_ptr = input.raw_ptr(((b*this->IN_CHANNEL + ic)*height + ih)*width + (ow + kw));
                                const float* w_ptr  = this->weights.raw_ptr(((oc*this->IN_CHANNEL + ic)*this->KERNEL_HEIGHT + kh)*this->KERNEL_WIDTH + kw);

                                avx2::matrix_t::mul(in_ptr, w_ptr, tmp, rem);

                                for (int t = 0; t < rem; ++t) sum += tmp[t];
                                kw += rem;
                            }
                        }
                    }
                    output.at(b, oc, oh, ow) = sum;
                }
            }
        }
    }
    return output;
}

tensor MindKernel::backward(const tensor &input, const tensor &grad_out) {
    const int batch = input.batch();
    const int height = input.height();
    const int width = input.width();

    const int out_height = height - this->KERNEL_HEIGHT + 1;
    const int out_width = width - this->KERNEL_WIDTH + 1;

    tensor grad_in(batch, this->IN_CHANNEL, height, width, 0.0f);
    tensor grad_weight(this->OUT_CHANNEL, this->IN_CHANNEL, this->KERNEL_HEIGHT, this->KERNEL_WIDTH, 0.0f);

    alignas(32) float tmp_a[8], tmp_b[8], tmp_res[8];

    for (int oc = 0; oc < this->OUT_CHANNEL; ++oc) {
        for (int ic = 0; ic < this->IN_CHANNEL; ++ic) {
            for (int kh = 0; kh < this->KERNEL_HEIGHT; ++kh) {
                for (int kw = 0; kw < this->KERNEL_WIDTH; ++kw) {
                    float sum = 0.0f;

                    for (int b = 0; b < batch; ++b) {
                        for (int oh = 0; oh < out_height; ++oh) {
                            const int ih = oh + kh;

                            int ow_flag = 0;
                            while (ow_flag < out_width) {
                                const int rem = std::min(8, out_width - ow_flag);

                                for (int t = 0; t < rem; ++t) {
                                    tmp_a[t] = input.at(b, ic, ih, ow_flag + t);
                                    tmp_b[t] = grad_out.at(b, oc, oh, ow_flag + t);
                                }

                                avx2::matrix_t::mul(tmp_a, tmp_b, tmp_res, rem);

                                for (int t = 0; t < rem; ++t) sum += tmp_res[t];

                                ow_flag += rem;
                            }
                        }
                    }
                    grad_weight.at(oc, ic, kh, kw) = sum;
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int ic = 0; ic < this->IN_CHANNEL; ++ic) {
            for (int ih = 0; ih < height; ++ih) {
                for (int iw = 0; iw < width; ++iw) {
                    float sum = 0.0f;

                    for (int oc = 0; oc < this->OUT_CHANNEL; ++oc) {
                        for (int kh = 0; kh < this->KERNEL_HEIGHT; ++kh) {
                            const int oh = ih - kh;
                            if (oh < 0 || oh >= out_height) continue;

                            int kw_flag = 0;
                            while (kw_flag < this->KERNEL_WIDTH) {
                                const int rem = std::min(8, this->KERNEL_WIDTH - kw_flag);

                                for (int t = 0; t < rem; ++t) {
                                    const int ow = iw - kw_flag + t;
                                    if (ow < 0 || ow >= out_width) continue;

                                    tmp_a[t] = this->weights.at(oc, ic, kh, kw_flag + t);
                                    tmp_b[t] = grad_out.at(b, oc, oh, ow);
                                }

                                avx2::matrix_t::mul(tmp_a, tmp_b, tmp_res, rem);

                                for (int t = 0; t < rem; ++t) sum += tmp_res[t];

                                kw_flag += rem;
                            }
                        }
                    }
                    grad_in.at(b, ic, ih, iw) = sum;
                }
            }
        }
    }

    return grad_in;
}
