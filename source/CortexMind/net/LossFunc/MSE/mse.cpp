//
// Created by muham on 9.12.2025.
//

#include "CortexMind/net/LossFunc/MSE/mse.hpp"
#include <CortexMind/framework/ErrorSystem/debug.hpp>

using namespace cortex::_fw;
using namespace cortex::net;
using namespace cortex;

tensor MeanSquared::forward(const tensor &predictions, const tensor &targets) {
    if (predictions.shape() != targets.shape()) {
        SynapticNode::captureFault(true, "cortex::net::MeanSquared::forward()", "Predictions and targets must have the same shape.");
    }

    const auto size = predictions.size();

    if (size == 0) {
        auto output = tensor(1, 1, 1, 1);
        output.fill(0);
        return output;
    }

    const tensor diff = predictions - targets;
    const tensor squaredDiff = diff * diff;

    float sum = 0;
    const float* data = squaredDiff.data().data()->data();
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }

    const float loss = sum / static_cast<float>(size);
    tensor output(1, 1, 1, 1);
    output.at(0, 0, 0, 0) = loss;
    return output;
}

tensor MeanSquared::backward(const tensor &predictions, const tensor &targets) {
    if (predictions.shape() != targets.shape()) {
        SynapticNode::captureFault(true, "cortex::net::MeanSquared::forward()", "Predictions and targets must have the same shape.");
    }

    const auto size = predictions.size();

    if (size == 0) {
        auto output = tensor(1, 1, 1, 1);
        output.fill(0.0);
        return output;
    }

    const tensor difference = predictions - targets;

    const float scale = 2 / static_cast<float>(size);

    tensor output = difference * scale;

    return output;
}
