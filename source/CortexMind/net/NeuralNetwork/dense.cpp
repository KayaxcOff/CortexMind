//
// Created by muham on 21.04.2026.
//

#include "CortexMind/net/NeuralNetwork/dense.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Dense::Dense(const int64 in, const int64 out, const sys::deviceType d_type) : LayerBase("Dense"), kInputDimension(in), kOutputDimension(out) {
    //this->kWeight = tensor({this->kInputDimension, this->kOutputDimension}, d_type, true);
    //this->kBias = tensor({this->kOutputDimension}, d_type, true);

    this->kWeight.rand();
    this->kBias.zero();
}

Dense::~Dense() = default;

tensor Dense::forward(tensor &input) {
    return input.dot(this->kWeight) + this->kBias;
}

std::vector<tensor> Dense::getWeights() {
    return {this->kWeight, this->kBias};
}

std::vector<tensor> Dense::getGradients() {
    return {this->kWeight.grad(), this->kBias.grad()};
}