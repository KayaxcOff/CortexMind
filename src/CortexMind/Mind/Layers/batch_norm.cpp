//
// Created by muham on 10.11.2025.
//

#include "CortexMind/Mind/NeuralNetwork/batch_norm.hpp"

#include <cmath>

using namespace cortex::nn;

BatchNorm::BatchNorm(const float64 _eps) : eps(_eps), gamma({}, {}), beta({}, {}), grad_gamma({}, {}), grad_beta({}, {}), x_norm({}, {}), mean({}), var({}) {}

BatchNorm::~BatchNorm() = default;

cortex::tensor BatchNorm::forward(const tensor &input) {
    const size N = input.get_rows();
    const size F = input.get_cols();

    if (this->gamma.get_rows() == 0) {
        this->gamma = tensor(1, F, 1.0);
        this->beta = tensor(1, F, 0.0);

        this->grad_gamma = tensor(1, F, 0.0);
        this->grad_beta = tensor(1, F, 0.0);
    }

    this->mean.assign(F, 0.0);
    this->var.assign(F, 0.0);

    for (size i = 0; i < F; ++i) {
        for (size j = 0; j < N; ++j) {
            this->mean[i] += input(j, i);
        }
        this->mean[i] /= static_cast<float64>(N);
    }

    for (size i = 0; i < N; ++i) {
        for (size j = 0; j < F; ++j) {
            this->var[i] += input(j, i);
        }
        this->var[i] /= static_cast<float64>(N);
    }

    this->x_norm = tensor(N, F);
    tensor output(N, F);

    for (size i = 0; i < N; ++i) {
        for (size j = 0; j < F; ++j) {
            const float64 inv_std = 1.0 / std::sqrt(this->var[j] + this->eps);

            this->x_norm(i, j) = (input(i, j) - this->mean[j]) * inv_std;
            output(i, j) = this->gamma(0, j) * this->x_norm(i, j) + this->beta(0, j);
        }
    }

    return output;
}

cortex::tensor BatchNorm::backward(const tensor &grad_output) {
    const size N = grad_output.get_rows();
    const size F = grad_output.get_cols();

    tensor grad_input(N, F);

    for (size i = 0; i < F; ++i) {
        this->grad_gamma(0, i) = 0.0;
        this->grad_beta(0, i) = 0.0;
    }

    for (size i = 0; i < N; ++i) {
        for (size j = 0; j < F; ++j) {
            this->grad_gamma(0, j) += grad_output(i, j) * this->x_norm(i, j);
            this->grad_beta(0, j) += grad_output(i, j);
        }
    }

    for (size j = 0; j < F; ++j) {
        const float64 inv_std = 1.0 / std::sqrt(this->var[j] + this->eps);

        float64 sum_dy = 0.0;
        float64 sum_dy_xn = 0.0;

        for (size i = 0; i < N; ++i) {
            sum_dy += grad_output(i, j);
            sum_dy_xn += grad_output(i, j) * this->x_norm(i, j);
        }

        for (size i = 0; i < N; ++i) {
            const float64 dy = grad_output(i, j);
            const float64 dx = (1.0 / N) * this->gamma(0, j) * inv_std * (N * dy - sum_dy - this->x_norm(i, j) * sum_dy_xn);

            grad_input(i, j) = dx;
        }
    }

    return grad_input;
}

cortex::tensor BatchNorm::getParams() const {
    tensor params(2, this->gamma.get_cols());

    for (size i = 0; i < this->gamma.get_cols(); ++i) {
        params(0, i) = this->gamma(0, i);
        params(1, i) = this->beta(0, i);
    }

    return params;
}

cortex::tensor BatchNorm::getGrads() const {
    tensor grads(2, this->grad_gamma.get_cols());

    for (size i = 0; i < this->grad_gamma.get_cols(); ++i) {
        grads(0, i) = this->grad_gamma(0, i);
        grads(1, i) = this->grad_beta(0, i);
    }

    return grads;
}

std::string BatchNorm::get_config() const {
    return "Batch Normalization";
}