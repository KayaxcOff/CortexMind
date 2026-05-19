//
// Created by muham on 19.05.2026.
//

#include "CortexMind/net/NeuralNetwork/dense.hpp"
#include <cmath>
#include <string>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Dense::Dense(int64 in_dim, int64 out_dim, const DeviceType device) : LayerBase("Dense (" + std::to_string(in_dim) + ", " + std::to_string(out_dim) + ")") {
    this->weight = tensor({in_dim, out_dim}, device, true);
    this->bias = tensor({1, out_dim}, device, true);

    float64 limit = std::sqrt(6.0 / (static_cast<float64>(in_dim) + static_cast<float64>(out_dim)));

    this->weight.uniform(-static_cast<float32>(limit), static_cast<float32>(limit));
    this->bias.zero();
}

Dense::~Dense() = default;

tensor Dense::forward(const tensor &input) {
    return input.matmul(this->weight) + this->bias;
}

std::vector<ref<tensor>> Dense::getParameters() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Dense::getGradients() {
    return {this->weight.grad(), this->bias.grad()};
}