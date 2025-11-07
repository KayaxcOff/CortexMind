//
// Created by muham on 4.11.2025.
//

#include "CortexMind/Model/Loss/MAE/mae.hpp"

using namespace cortex::loss;
using namespace cortex;

double MeanAbsolute::forward(const math::MindVector &predictions, const math::MindVector &targets) {
    double loss = 0.0;

    for (int i = 0; i < predictions.size(); ++i) {
        loss += std::abs(targets[i] - predictions[i]);
    }
    return loss / static_cast<double>(predictions.size());
}

math::MindVector MeanAbsolute::backward(const math::MindVector &predictions, const math::MindVector &targets) {
    math::MindVector grad(predictions.size());
    const double scale = 1.0 / static_cast<double>(predictions.size());

    for (size_t i = 0; i < predictions.size(); ++i) {
        const double diff = predictions[i] - targets[i];
        grad[i] = scale * (diff > 0 ? 1.0 : (diff < 0 ? -1.0 : 0.0));
    }

    return grad;
}