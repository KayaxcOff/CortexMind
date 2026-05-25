//
// Created by muham on 25.05.2026.
//

#include "CortexMind/net/NeuralNetwork/batch_norm.hpp"
#include <string>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

BatchNormalization::BatchNormalization(const std::vector<int64>& axes, const int64 num_features, const float32 momentum, const float32 epsilon, const DeviceType device) : LayerBase("BatchNorm(" + std::to_string(num_features) + ")") {
    this->num_features = num_features;
    this->momentum = momentum;
    this->epsilon = epsilon;

    this->axes = axes;

    this->gamma = Tensor({1, num_features}, device, true);
    this->beta = Tensor({1, num_features}, device, true);

    this->gamma.ones();
    this->beta.zero();

    this->running_mean = Tensor({1, num_features}, device, false);
    this->running_var = Tensor({1, num_features}, device, false);

    this->running_mean.zero();
    this->running_var.ones();
}

BatchNormalization::~BatchNormalization() = default;

tensor BatchNormalization::forward(const tensor &input) {
    CXM_ASSERT(input.shape().back() != this->num_features, "Input last dimension must match num_features");

    tensor output;

    if (this->flag()) {
        tensor mean = input.mean(this->axes, true);

        const tensor diff = input - mean;
        tensor var = (diff * diff).mean(this->axes, true);

        tensor x_norm = diff * (var + this->epsilon).rsqrt();
        output = this->gamma * x_norm + this->beta;

        this->running_mean = (1.0f - this->momentum) * this->running_mean + this->momentum * mean;
        this->running_var = (1.0f - this->momentum) * this->running_var + this->momentum * var;

    } else {
        tensor x_norm = (input - this->running_mean) * (this->running_var + this->epsilon).rsqrt();
        output = this->gamma * x_norm + this->beta;
    }

    return output;
}

std::vector<ref<tensor>> BatchNormalization::getParameters() {
    return {this->gamma, this->beta};
}

std::vector<ref<tensor>> BatchNormalization::getGradients() {
    return {this->gamma.grad(), this->beta.grad()};
}