//
// Created by muham on 23.12.2025.
//

#include "CortexMind/net/ActivationFunc/Softmax/softmax.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <cmath>

using namespace cortex::_fw;
using namespace cortex::net;
using namespace cortex;

Softmax::Softmax() = default;

Softmax::~Softmax() = default;

tensor Softmax::forward(const tensor &input) {
    this->output.allocate(input.batch(), input.channel(), input.height(), input.width());

    const int batch = input.batch();
    const int num_classes = input.width();

    for (int i = 0; i < batch; ++i) {
        float maxValue = -std::numeric_limits<float>::infinity();
        for (int j = 0;j < num_classes; ++j) {
            maxValue = std::max(maxValue, input.at(i, 0, 0, j));
        }

        float sum = 0.0f;
        avx2::reg sum_vec = avx2::zero();

        int c = 0;
        while (c + 8 <= num_classes) {
            float vals[8];
            for (int j = 0; j < 8; ++j) {
                vals[j] = input.at(i, 0, 0, c + j) - maxValue;
            }

            const avx2::reg x_vec = avx2::load(vals);
            const avx2::reg exp_vec = avx2::exp_approx(x_vec);

            float exp_vals[8];
            avx2::store(exp_vals, exp_vec);
            for (int j = 0; j < 8; ++j) {
                this->output.at(i, 0, 0, c + j) = exp_vals[j];
            }

            sum_vec = avx2::add(sum_vec, exp_vec);
            c += 8;
        }

        sum += avx2::h_sum(sum_vec);

        while (c < num_classes) {
            const float exp_val = std::exp(input.at(i, 0, 0, c) - maxValue);
            this->output.at(i, 0, 0, c) = exp_val;
            sum += exp_val;
            c++;
        }

        const float inv_sum = 1.0f / sum;
        const avx2::reg inv_sum_vec = avx2::broadcast(inv_sum);

        c = 0;
        while (c + 8 <= num_classes) {
            float vals[8];
            for (int j = 0; j < 8; ++j) {
                vals[j] = this->output.at(i, 0, 0, c + j);
            }

            const avx2::reg val_vec = avx2::load(vals);
            const avx2::reg result_vec = avx2::mul(val_vec, inv_sum_vec);

            float results[8];
            avx2::store(results, result_vec);
            for (int j = 0; j < 8; ++j) {
                this->output.at(i, 0, 0, c + j) = results[j];
            }

            c += 8;
        }

        while (c < num_classes) {
            this->output.at(i, 0, 0, c) *= inv_sum;
            c++;
        }
    }
    return this->output;
}

tensor Softmax::backward(const tensor &grad_output) {
    const int batch = grad_output.batch();
    const int num_classes = grad_output.width();

    tensor grad_input(batch, 1, 1, num_classes);

    for (int b = 0; b < batch; ++b) {
        float dot_product = 0.0f;
        avx2::reg dot_vec = avx2::zero();

        int c = 0;
        while (c + 8 <= num_classes) {
            float softmax_vals[8], grad_vals[8];
            for (int i = 0; i < 8; ++i) {
                softmax_vals[i] = this->output.at(b, 0, 0, c + i);
                grad_vals[i] = grad_output.at(b, 0, 0, c + i);
            }

            const avx2::reg s_vec = avx2::load(softmax_vals);
            const avx2::reg g_vec = avx2::load(grad_vals);
            const avx2::reg prod = avx2::mul(s_vec, g_vec);

            dot_vec = avx2::add(dot_vec, prod);
            c += 8;
        }

        dot_product += avx2::h_sum(dot_vec);

        while (c < num_classes) {
            dot_product += this->output.at(b, 0, 0, c) * grad_output.at(b, 0, 0, c);
            c++;
        }

        const avx2::reg dot_broadcast = avx2::broadcast(dot_product);

        c = 0;
        while (c + 8 <= num_classes) {
            float softmax_vals[8], grad_vals[8];
            for (int i = 0; i < 8; ++i) {
                softmax_vals[i] = this->output.at(b, 0, 0, c + i);
                grad_vals[i] = grad_output.at(b, 0, 0, c + i);
            }

            const avx2::reg s_vec = avx2::load(softmax_vals);
            const avx2::reg g_vec = avx2::load(grad_vals);
            const avx2::reg diff = avx2::sub(g_vec, dot_broadcast);
            const avx2::reg result = avx2::mul(s_vec, diff);

            float results[8];
            avx2::store(results, result);
            for (int i = 0; i < 8; ++i) {
                grad_input.at(b, 0, 0, c + i) = results[i];
            }

            c += 8;
        }

        while (c < num_classes) {
            grad_input.at(b, 0, 0, c) = this->output.at(b, 0, 0, c) * (grad_output.at(b, 0, 0, c) - dot_product);
            c++;
        }
    }

    return grad_input;
}