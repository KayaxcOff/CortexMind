//
// Created by muham on 1.03.2026.
//

#include "CortexMind/net/LossFunctions/mse.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::loss;
using namespace cortex::_fw;
using namespace cortex;

MeanSquared::MeanSquared() : Loss("MSE") {}

MeanSquared::~MeanSquared() = default;

tensor MeanSquared::forward(const tensor &predicted, const tensor &target) {
    CXM_ASSERT(predicted.shape() == target.shape(), "cortex::net::MeanSquared::forward", "Shape mismatch");

    this->last_diff = predicted - target;
    this->last_sq   = this->last_diff * this->last_diff;

    return this->last_sq.sum();
}