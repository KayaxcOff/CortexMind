//
// Created by muham on 18.03.2026.
//

#include "CortexMind/net/NeuralNetwork/dense.hpp"
#include <cmath>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Dense::Dense(const int64 in_feats, const int64 out_feats, const dev d) : Layer(true, "Dense"), INPUT_FEATS(in_feats), OUTPUT_FEATS(out_feats) {
    this->weight = tensor({this->INPUT_FEATS, this->OUTPUT_FEATS}, d, true);
    this->bias = tensor({1, this->OUTPUT_FEATS}, d, true);

    const float32 limit = std::sqrt(6.0f / static_cast<float32>(in_feats + out_feats));

    this->weight.uniform(-limit, limit);
    this->bias.zero();
}

Dense::~Dense() = default;

tensor Dense::forward(tensor &input) {
    return input.matmul(this->weight) + this->bias;
}

std::vector<ref<tensor>> Dense::parameters() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Dense::gradients() {
    return {this->weight.grad(), this->bias.grad()};
}