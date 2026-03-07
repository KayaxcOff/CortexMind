//
// Created by muham on 5.03.2026.
//

#include "CortexMind/net/NeuralNetwork/batch_norm.hpp"
#include <CortexMind/core/Graph/flow_ops.hpp>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

BatchNorm1D::BatchNorm1D(const int64 num_features, const float32 eps, const float32 momentum) : Layer(true, "BatchNorm1D"), features(num_features), momentum(momentum), eps(eps) {

    this->gamma = tensor({this->features}, true);
    this->beta = tensor({this->features}, true);

    this->running_mean = tensor({this->features});
    this->running_var = tensor({this->features});

    this->gamma.ones();
    this->beta.zero();

    this->running_mean.zero();
    this->running_var.zero();
}

BatchNorm1D::~BatchNorm1D() = default;

tensor BatchNorm1D::forward(tensor &input) {
    this->last_input = input;

    if (this->flag) {
        auto mean = this->last_input.sum(0) / static_cast<float32>(this->last_input.shape()[0]);

        auto diff = this->last_input - mean;
        auto var = (diff.pow(2)).sum(0) / static_cast<float32>(input.shape()[0]);
        auto norm = diff / (var + this->eps).sqrt();
        auto output = this->gamma * norm + this->beta;

        this->running_mean = this->momentum * mean + (1 - this->momentum) * this->running_mean;
        this->running_var  = this->momentum * var  + (1 - this->momentum) * this->running_var;

        return output;
    }
    auto norm = (input - this->running_mean) / (this->running_var + this->eps).sqrt();

    tensor output = this->gamma * norm + this->beta;

    if (output.requires_grad()) {
        tensor X;
        auto flow = std::make_shared<meta::batch_norm>(&this->last_input, &this->gamma, &this->beta, &X, &this->running_mean, &this->running_var, this->eps);

        output.set_flow(std::move(flow));
    }

    return output;
}

std::vector<ref<tensor>> BatchNorm1D::parameters() {
    return {this->gamma, this->beta};
}

std::vector<ref<tensor>> BatchNorm1D::gradients() {
    return {this->gamma.grad(), this->beta.grad()};
}
