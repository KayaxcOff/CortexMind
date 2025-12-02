//
// Created by muham on 1.12.2025.
//

#include "CortexMind/net/NeuralNetwork/BatchNorm/batch_norm.hpp"
#include <CortexMind/framework/Log/log.hpp>
#include <CortexMind/utils/MathTools/pch.hpp>
#include <stdexcept>

using namespace cortex::nn;
using namespace cortex;

BatchNorm::BatchNorm(const size_t num_feat, const float eps, const float momentum) : gamma(0, 0, 0), beta(0, 0, 0), grad_gamma(0, 0, 0), grad_beta(0, 0, 0), running_mean(0, 0, 0), running_var(0, 0, 0), momentum(momentum), eps(eps), num_feats(num_feat), is_training(true), cached_input(0, 0, 0), cached_norm_input(0, 0,0), cached_variance(0, 0, 0), cached_mean(0, 0, 0) {
    this->gamma = tensor(1, 1, this->num_feats, true);
    this->beta = tensor(1, 1, this->num_feats, true);

    this->grad_gamma = tensor(1, 1, this->num_feats, false);
    this->grad_beta = tensor(1, 1, this->num_feats, false);

    this->running_mean = tensor(1, 1, this->num_feats, false);
    this->running_var = tensor(1, 1, this->num_feats, false);

    this->gamma.fill(1.0);
    this->beta.fill(0.0);
    this->running_mean.fill(0.0);
    this->running_var.fill(0.0);
}

void BatchNorm::train() {
    this->is_training = true;
}

void BatchNorm::eval() {
    this->is_training = false;
}

tensor BatchNorm::forward(tensor &input) {
    const size_t B = input.get_shape()[0];
    const size_t R = input.get_shape()[1];
    const size_t C = input.get_shape()[2];

    if (C != this->num_feats) {
        log("BatchNorm forward: input feature size does not match num_feats");
        throw std::runtime_error("BatchNorm forward: input feature size does not match num_feats");
    }

    tensor mean_tensor(1, 1, C, false);
    tensor var_tensor(1, 1, C, false);

    if (this->is_training) {
        const size_t N = B * R;

        for (size_t f = 0; f < C; ++f) {
            double sum = 0.0;
            for (size_t b = 0; b < B; ++b) {
                for (size_t r = 0; r < R; ++r) {
                    sum += input(b, r, f);
                }
            }
            double mu = sum / static_cast<double>(N);
            mean_tensor(0, 0, f) = mu;

            double sum_squares = 0.0;
            for (size_t b = 0; b < B; ++b) {
                for (size_t r = 0; r < R; ++r) {
                    double diff = input(b, r, f) - mu;
                    sum_squares += diff * diff;
                }
            }
            double var = sum_squares / static_cast<double>(N);
            var_tensor(0, 0, f) = var;
        }
        this->cached_input = mean_tensor;
        this->cached_variance = var_tensor;

        tensor momentum_tensor(1, 1, 1, false);
        momentum_tensor.fill(this->momentum);
        tensor one_minus_momentum(1, 1, 1, false);
        one_minus_momentum.fill(1.0 - this->momentum);

        this->running_mean = (one_minus_momentum * this->running_mean) + (momentum_tensor * mean_tensor);
        this->running_var = (one_minus_momentum * this->running_var) + (momentum_tensor * var_tensor);
    } else {
        mean_tensor = this->running_mean;
        var_tensor = this->running_var;
    }

    tensor stddev = exp(var_tensor * 0.5f) + this->eps;
    tensor centered_input = input - mean_tensor;

    tensor normalized_input = centered_input / stddev;
    this->cached_norm_input = normalized_input;

    tensor output = (this->gamma * normalized_input) + this->beta;

    return output;
}

tensor BatchNorm::backward(tensor &grad_output) {
    if (!this->is_training) {
        log("BatchNorm backward: input feature size is not correct");
        throw std::runtime_error("BatchNorm backward: input feature size is not correct");
    }

    this->grad_gamma.zero();
    this->grad_beta.zero();

    return grad_output;
}

std::vector<tensor*> BatchNorm::getParameters() {
    return {&this->gamma, &this->beta};
}

std::vector<tensor*> BatchNorm::getGradients() {
    return {&this->grad_gamma, &this->grad_beta};
}

std::string BatchNorm::config() {
    return "Batch Normalization";
}