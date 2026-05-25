//
// Created by muham on 25.05.2026.
//

#include "CortexMind/net/LossFunction/bce.hpp"
#include <string>

using namespace cortex::_fw;
using namespace cortex::loss;
using namespace cortex;

BinaryCrossEntropy::BinaryCrossEntropy(const float32 eps) : LossBase("BCE(" + std::to_string(eps) + ")") {
    this->eps = eps;
}

BinaryCrossEntropy::~BinaryCrossEntropy() = default;

tensor BinaryCrossEntropy::forward(const tensor &predict, const tensor &target) {
    return predict;
}