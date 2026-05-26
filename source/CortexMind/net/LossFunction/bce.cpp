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
    const tensor p = predict.clamp(this->eps, 1.0f - this->eps);

    const tensor loss = (target * p.log() + (1.0f - target) * (1.0f - p).log()).neg();

    return loss.sum() / static_cast<float32>(loss.len());
}