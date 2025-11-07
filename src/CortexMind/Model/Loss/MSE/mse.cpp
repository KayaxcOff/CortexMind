//
// Created by muham on 3.11.2025.
//

#include "CortexMind/Model/Loss/MSE/mse.hpp"
#include <stdexcept>

using namespace cortex::loss;
using namespace cortex;

MeanSquared::MeanSquared() = default;

double MeanSquared::forward(const math::MindVector &predictions, const math::MindVector &targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("MeanSquared::forward -> size mismatch between predictions and targets");
    }


    double loss = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        const double diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    return loss / static_cast<double>(predictions.size());
}

math::MindVector MeanSquared::backward(const math::MindVector &predictions, const math::MindVector &targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("MeanSquared::backward -> size mismatch between predictions and targets");
    }

    math::MindVector grad(targets.size());
    const double scale = 2.0 / static_cast<double>(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        grad[i] = scale * (predictions[i] - targets[i]);
    }
    return grad;
}