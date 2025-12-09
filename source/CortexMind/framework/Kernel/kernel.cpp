//
// Created by muham on 7.12.2025.
//

#include "CortexMind/framework/Kernel/kernel.hpp"
#include <CortexMind/framework/AVX2/avx2.hpp>

using namespace cortex::_fw;
using namespace cortex;

ConvKernel::ConvKernel(const int out_c, const int in_c, const int k_h, const int k_w, const float initValue) : weights(out_c, in_c, k_h, k_w, initValue) {}

ConvKernel::~ConvKernel() = default;

tensor ConvKernel::apply(const tensor &input) {
    const int B = input.shapeIdx(0);
    //const int C = input.shapeIdx(1);
    const int H = input.shapeIdx(2);
    const int W = input.shapeIdx(3);

    const int out_c = this->weights.shapeIdx(0);
    const int in_c  = this->weights.shapeIdx(1);
    const int k_h   = this->weights.shapeIdx(2);
    const int k_w   = this->weights.shapeIdx(3);

    const int out_h = H - k_h + 1;
    const int out_w = W - k_w + 1;

    tensor output(B, out_c, out_h, out_w, 0.0f);
    alignas(32) float tmp[8];

    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;

                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < k_h; ++kh) {
                            const int ih = oh + kh;

                            int kw = 0;
                            while (kw < k_w) {
                                const int rem = std::min(8, k_w - kw);
                                const float* inp_ptr = &input.at(b, ic, ih, ow + kw);
                                const float* w_ptr   = &this->weights.at(oc, ic, kh, kw);

                                avx::mul_kernel(inp_ptr, w_ptr, tmp, rem);

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

tensor ConvKernel::backward(const tensor &in, tensor &out) {
    const int batch = in.shapeIdx(0);
    const int in_c = in.shapeIdx(1);
    const int in_h = in.shapeIdx(2);
    const int in_w = in.shapeIdx(3);

    const int out_c = this->weights.shapeIdx(0);
    const int k_h = this->weights.shapeIdx(2);
    const int k_w = this->weights.shapeIdx(3);

    const int out_h = in_h - k_h + 1;
    const int out_w = in_w - k_w + 1;

    tensor grad_in(batch, in_c, in_h, in_w, 0.0f);
    tensor grad_weights(out_c, in_c, k_h, k_w, 0.0f);

    alignas(32) float tmp_a[8];
    alignas(32) float tmp_b[8];
    alignas(32) float tmp_res[8];

    for (int i = 0; i < out_c; ++i) {
        for (int j = 0; j < in_c; ++j) {
            for (int k = 0; k < k_h; ++k) {
                for (int l = 0; l < k_w; ++l) {
                    float sum = 0.0f;
                    for (int m = 0; m < batch; ++m) {
                        for (int n = 0; n < out_h; ++n) {
                            const int ih = n + k;
                            int ow = 0;
                            while (ow < out_w) {
                                const int rem = std::min(8, out_w - ow);

                                for (int t = 0; t < rem; ++t) {
                                    tmp_a[t] = in.at(i, j, ih, ow + t);
                                    tmp_b[t] = out.at(i, i, n, ow + t);
                                }
                                avx::mul_kernel(tmp_a, tmp_b, tmp_res, rem);

                                for (int t = 0; t < rem; ++t) sum += tmp_res[t];

                                ow += rem;
                            }
                        }
                    }
                    grad_weights.at(i, j, k, l) = sum;
                }
            }
        }
    }

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < in_c; ++j) {
            for (int k = 0; k < in_h; ++k) {
                for (int l = 0; l < in_w; ++l) {
                    float sum = 0.0f;
                    for (int m = 0; m < out_c; ++m) {
                        for (int n = 0; n < k_h; ++n) {
                            for (int o = 0; o < k_w; ++o) {
                                const int oh = k - m;
                                int ow = l - n;

                                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                    tmp_a[0] = out.at(i, m, oh, ow);
                                    tmp_b[0] = this->weights.at(m, j, n, m);
                                    avx::mul_kernel(tmp_a, tmp_b, tmp_res, 1);
                                    sum += tmp_res[0];
                                }
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