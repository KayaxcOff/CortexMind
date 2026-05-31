//
// Created by muham on 29.05.2026.
//

#include "CortexMind/net/LossFunction/cce.hpp"
#include <string>

using namespace cortex::_fw;
using namespace cortex::loss;
using namespace cortex;

CategoricalCrossEntropy::CategoricalCrossEntropy(const float32 eps) : LossBase("CCE(" + std::to_string(eps) + ")") {
    this->eps = eps;
}

CategoricalCrossEntropy::~CategoricalCrossEntropy() = default;

tensor CategoricalCrossEntropy::forward(const tensor &predict, const tensor &target) {
    const tensor probs = predict.clamp(this->eps, 1.0f - this->eps);

    const tensor output = target.mul(probs.log()).sum({1}, false).neg();

    return output / static_cast<float32>(output.len());
}