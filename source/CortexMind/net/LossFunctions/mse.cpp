//
// Created by muham on 5.05.2026.
//

#include "CortexMind/net/LossFunctions/mse.hpp"

using namespace cortex::loss;
using namespace cortex::_fw;
using namespace cortex;

MeanSquared::MeanSquared() : LossBase("MSE") {}

MeanSquared::~MeanSquared() = default;

tensor MeanSquared::forward(tensor &prediction, tensor &target) {
    return (prediction - target).pow(2).sum() / static_cast<float32>(prediction.len());
}