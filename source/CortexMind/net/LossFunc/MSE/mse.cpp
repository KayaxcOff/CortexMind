//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/LossFunc/MSE/mse.hpp"
#include <CortexMind/framework/Log/log.hpp>
#include <stdexcept>

using namespace cortex::loss;
using namespace cortex;

tensor MeanSquared::forward(const tensor &predictions, const tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
        log("Predictions and targets must have the same shape.");
        throw std::invalid_argument("Predictions and targets must have the same shape.");
    }

    const size_t total_size = predictions.get_data().size();
    if (total_size == 0) {
        auto output = tensor(1, 1, 1);
        output.fill(0.0);
        return output;
    }

    const tensor diff = predictions - targets;
    tensor squaredDiff = diff * diff;

    double squaredSum = 0;
    for (const double val : squaredDiff.get_data()) {
        squaredSum += val;
    }

    const double loss = squaredSum / static_cast<double>(total_size);
    tensor output(1, 1, 1);
    output(0, 0, 0) = loss;
    return output;
}

tensor MeanSquared::backward(const tensor &predictions, const tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
        log("Predictions and targets must have the same shape.");
        throw std::invalid_argument("Predictions and targets must have the same shape.");
    }

    const size_t total_size = predictions.get_data().size();

    if (total_size == 0) {
        auto output = tensor(1, 1, 1);
        output.fill(0.0);
        return output;
    }

    const tensor difference = predictions - targets;

    const double scale = 2.0 / static_cast<double>(total_size);

    tensor output = difference * scale;

    return output;
}
