//
// Created by muham on 25.02.2026.
//

#include "CortexMind/net/NeuralNetwork/dense.hpp"
#include <cmath>
#include <string>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Dense::Dense(const i64 in_feats, const i64 out_feats, const sys::device _dev) : Layer(true, "Dense(" + std::to_string(in_feats) + ", " + std::to_string(out_feats) + ")"), INPUT_FEATURES(in_feats), OUTPUT_FEATURES(out_feats) {
    this->weight = tensor({this->INPUT_FEATURES, this->OUTPUT_FEATURES}, _dev, true);
    this->bias = tensor({1, this->OUTPUT_FEATURES}, _dev, true);

    const float32 limit = std::sqrt(6.0f / static_cast<float32>(in_feats + out_feats));

    this->weight.uniform_rand(-limit, limit);
    this->bias.zero();
}

Dense::~Dense() = default;

tensor Dense::forward(tensor& input) {
    this->last_input = input;
    this->last_input.clear_flow();
    this->last_z = this->last_input.matmul(this->weight);
    return this->last_z + this->bias;
}

std::vector<ref<tensor>> Dense::parameters() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Dense::gradients() {
    return {this->weight.grad(), this->bias.grad()};
}