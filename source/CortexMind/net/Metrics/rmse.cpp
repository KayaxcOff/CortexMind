//
// Created by muham on 31.05.2026.
//

#include "CortexMind/net/Metrics/rmse.hpp"

using namespace cortex::_fw;
using namespace cortex::metric;
using namespace cortex;

RootMeanSquared::RootMeanSquared() : MetricBase("RMSE") {}

RootMeanSquared::~RootMeanSquared() = default;

float32 RootMeanSquared::forward(const tensor &predict, const tensor &target) {
    const tensor diff = predict - target;
    const tensor sq = diff.pow(2.0f);

    tensor output = (sq.sum() / static_cast<float32>(sq.len())).sqrt();

    return output.get()[0];
}