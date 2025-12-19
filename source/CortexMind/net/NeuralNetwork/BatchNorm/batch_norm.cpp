//
// Created by muham on 16.12.2025.
//

#include "CortexMind/net/NeuralNetwork/BatchNorm/batch_norm.hpp"
#include <cmath>

using namespace cortex::nn;
using namespace cortex;

BatchNorm::BatchNorm(const float epsilon, const float momentum) : momentum(momentum), eps(epsilon) {}

BatchNorm::~BatchNorm() = default;

tensor BatchNorm::forward(const tensor &input) {
    this->gamma.allocate(1, input.channel(), 1, 1);
    this->gamma.fill(1.0f);

    this->beta.allocate(1, input.channel(), 1, 1);
    this->beta.zero();

    this->running_mean.allocate(1, input.channel(), 1, 1);
    this->running_mean.zero();

    this->running_var.allocate(1, input.channel(), 1, 1);
    this->running_var.zero();

    tensor output(input.batch(), input.channel(), input.height(), input.width());

    this->x_hat.allocate(input.batch(), input.channel(), input.height(), input.width());
    this->batch_mean.allocate(1, input.channel(), 1, 1);
    this->batch_var.allocate(1, input.channel(), 1, 1);

    for (int i = 0; i < input.channel(); ++i) {
        float mean = 0;
        float var = 0;
        for (int j = 0; j < input.batch(); ++j) {
            for (int k = 0; k < input.width(); ++k) {
                for (int l = 0; l < input.height(); ++l) {
                    mean += input.at(j, i, l, k);
                }
            }
        }
        mean /= static_cast<float>(input.batch() * input.height() * input.width());

        for (int j = 0; j < input.batch(); ++j) {
            for (int k = 0; k < input.width(); ++k) {
                for (int l = 0; l < input.height(); ++l) {
                    var += (input.at(j, i, l, k) - mean) * (input.at(j, i, l, k) - mean);
                }
            }
        }
        var /= static_cast<float>(input.batch() * input.height() * input.width());

        this->batch_mean.at(0, i, 0, 0) = mean;
        this->batch_var.at(0, i, 0, 0) = var;

        this->running_mean.at(0, i, 0, 0) = this->momentum * this->running_mean.at(0, i, 0, 0) + (1 - this->momentum) * mean;
        this->running_var.at(0, i, 0, 0) = this->momentum * this->running_var.at(0, i, 0, 0) + (1 - this->momentum) * var;

        const float inv_std = 1.0f / std::sqrt(var + this->eps);

        for (int j = 0; j < input.batch(); ++j) {
            for (int k = 0; k < input.width(); ++k) {
                for (int l = 0; l < input.height(); ++l) {
                    const float xn = (input.at(i, j, l, k) - mean) * inv_std;
                    this->x_hat.at(j, i, l, k) = xn;
                    output.at(j, i, l, k) = this->gamma.at(0, i, 0, 0) * xn + this->beta.at(0, i, 0, 0);
                }
            }
        }
    }
    return output;
}

tensor BatchNorm::backward(const tensor &grad_output) {
    tensor grad_input(grad_output.batch(), grad_output.channel(), grad_output.height(), grad_output.width());

    for (int i = 0; i < grad_output.channel(); ++i) {
        const float gamma_c = this->gamma.at(0, i, 0, 0);
        const float var_c = this->batch_var.at(0, i, 0, 0);
        const float inv_std = 1.0f / std::sqrt(var_c + this->eps);

        float d_gamma = 0.0f, d_beta = 0.0f;
        for (int j = 0; j < grad_output.batch(); ++j) {
            for (int k = 0; k < grad_output.width(); ++k) {
                for (int l = 0; l < grad_output.height(); ++l) {
                    const float d = grad_output.at(j, i, l, k);
                    const float xh = this->x_hat.at(j, i, l, k);
                    d_gamma += d * xh;
                    d_beta += d;
                }
            }
        }

        for (int j = 0; j < grad_output.batch(); ++j) {
            for (int k = 0; k < grad_output.width(); ++k) {
                for (int l = 0; l < grad_output.height(); ++l) {
                    const float d = grad_output.at(j, i, l, k);
                    const float xh = this->x_hat.at(j, i, l, k);
                    grad_input.at(j, i, l, k) = gamma_c * inv_std * (d - d_beta / (grad_output.batch() * grad_output.height() * grad_output.width()) - xh * d_gamma / (grad_output.batch() * grad_output.height() * grad_output.width()));
                }
            }
        }
    }
    return grad_input;
}

std::string BatchNorm::config() {
    return "BatchNorm";
}
