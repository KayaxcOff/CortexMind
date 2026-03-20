//
// Created by muham on 19.03.2026.
//

#include "CortexMind/net/LossFunctions/mse.hpp"

using namespace cortex::loss;
using namespace cortex::_fw;
using namespace cortex;

MeanSquared::MeanSquared() : Loss("MSE") {}

MeanSquared::~MeanSquared() = default;

tensor MeanSquared::forward(tensor &predicted, tensor &target) {
    return (predicted - target).pow().sum() / static_cast<float32>(predicted.numel());
}