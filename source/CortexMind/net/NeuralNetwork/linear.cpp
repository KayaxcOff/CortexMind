//
// Created by muham on 25.04.2026.
//

#include "CortexMind/net/NeuralNetwork/linear.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Linear::Linear(const int64 input_dim, const int64 output_dim, const sys::deviceType d_type) : LayerBase("Linear"), input_dim(input_dim), output_dim(output_dim) {
    this->weight = tensor({this->input_dim, this->output_dim}, d_type, true);
    this->bias = tensor({this->input_dim, this->output_dim}, d_type, true);

    this->weight.rand();
    this->bias.zero();
}

Linear::~Linear() = default;

tensor Linear::forward(tensor &input) {
    return input.dot(this->weight) + this->bias;
}

std::vector<ref<tensor>> Linear::getWeight() {
    return {this->weight, this->bias};
}

std::vector<ref<tensor>> Linear::getGradient() {
    return {this->weight.grad(), this->bias.grad()};
}