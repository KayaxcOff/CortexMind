//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/LossFunc/MAE/mae.hpp"
#include <CortexMind/framework/Log/log.hpp>
#include <stdexcept>

using namespace cortex::loss;
using namespace cortex;

tensor MeanAbsolute::forward(const tensor &predictions, const tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
        log("Predictions and targets must have the same shape for MAE loss.");
        throw std::invalid_argument("Predictions and targets must have the same shape for MAE loss.");
    }

    const size_t totalSize = predictions.get_data().size();

    if (totalSize == 0) {
        auto output = tensor(1, 1, 1);
        output.fill(0.0);
        return output;
    }

    double sum = 0.0;
    for (size_t i = 0; i < totalSize; ++i) {
        sum += std::fabs(predictions.get_data()[i] - targets.get_data()[i]);
    }

    const double loss = sum / static_cast<double>(totalSize);

    tensor output(1, 1, 1, false);
    output(0, 0, 0) = loss;
    return output;
}

tensor MeanAbsolute::backward(const tensor &predictions, const tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
        log("Predictions and targets must have the same shape for MAE loss.");
        throw std::invalid_argument("Predictions and targets must have the same shape for MAE loss.");
    }


        const size_t totalSize = predictions.get_data().size();
        if (totalSize == 0) {
            auto output = tensor(1, 1, 1);
            output.fill(0.0);
            return output;
        }

        tensor output(predictions.get_shape()[0], predictions.get_shape()[1], predictions.get_shape()[2], false);
        const double scale = 1.0 / static_cast<double>(totalSize);

        for (size_t i = 0; i < totalSize; ++i) {
            const double sum = predictions.get_data()[i] - targets.get_data()[i];

            double sign = 0.0;
            if (sum > 1e-9) {
                sign = 1.0;
            } else if (sum < -1e-9) {
                sign = -1.0;
            }

            output.get_data()[i] = sign * scale;
        }
    return output;
}