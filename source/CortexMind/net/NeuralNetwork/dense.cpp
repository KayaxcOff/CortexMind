//
// Created by muham on 3.05.2026.
//

#include "CortexMind/net/NeuralNetwork/dense.hpp"
#include <cmath>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Dense::Dense(const int64 in_dim, const int64 out_dim, const sys::deviceType d_type) : LayerBase("Dense"), INPUT_DIM(in_dim), OUTPUT_DIM(out_dim) {
    this->weight = tensor({this->INPUT_DIM, this->OUTPUT_DIM}, d_type, true);
    this->bias = tensor({1, this->OUTPUT_DIM}, d_type, false);

    auto limit = static_cast<float32>(std::sqrt(6.0 / static_cast<float64>(this->INPUT_DIM + this->OUTPUT_DIM)));

    this->weight.rand(-limit, limit);
    this->bias.zero();
}

Dense::~Dense() = default;

tensor Dense::forward(tensor &input) {
    return input.dot(this->weight) + this->bias;
}

std::vector<ref<tensor>> Dense::getWeight() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Dense::getGradient() {
    return {this->weight.grad(), this->bias.grad()};
}