//
// Created by muham on 20.05.2026.
//

#include "CortexMind/net/LossFunction/mse.hpp"

using namespace cortex::_fw;
using namespace cortex::loss;
using namespace cortex;

MeanSquared::MeanSquared() : LossBase("MSE") {}

MeanSquared::~MeanSquared() = default;

tensor MeanSquared::forward(const tensor &predict, const tensor &target) {
    const tensor diff = predict - target;
    const tensor sq   = diff.pow(2.0f);

    return sq.sum() / static_cast<float32>(sq.len());
}