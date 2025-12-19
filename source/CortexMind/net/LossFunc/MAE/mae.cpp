//
// Created by muham on 19.12.2025.
//

#include "CortexMind/net/LossFunc/MAE/mae.hpp"

using namespace cortex::net;
using namespace cortex;

MeanAbsolute::MeanAbsolute() = default;

MeanAbsolute::~MeanAbsolute() = default;

tensor MeanAbsolute::forward(const tensor &predictions, const tensor &targets) const {
    tensor diff = targets - predictions;
    tensor output(targets.batch(), 1, 1, 1);

    for (int i = 0; i < targets.batch(); ++i) {
        float sum = 0.0f;
        for (int j = 0; j < targets.channel(); ++j) {
            for (int k = 0; k < targets.height(); ++k) {
                for (int l = 0; l < targets.width(); ++l) {
                    sum += std::abs(diff.at(i, j, k, l));
                }
            }
        }
        output.at(i, 0, 0, 0) = sum / static_cast<float>(targets.channel() * targets.height() * targets.width());
    }
    return output;
}

tensor MeanAbsolute::backward(const tensor &predictions, const tensor &targets) const {
    tensor grad(predictions.batch(), predictions.channel(), predictions.height(), predictions.width());

    for (int i = 0; i < predictions.batch(); ++i) {
        const float scale = 1.0f / static_cast<float>(targets.channel() * targets.height() * targets.width());
        for (int j = 0; j < predictions.channel(); ++j) {
            for (int k = 0; k < predictions.height(); ++k) {
                for (int l = 0; l < predictions.width(); ++l) {
                    float diff = predictions.at(i, j, k, l) - targets.at(i, j, k, l);
                    if (diff > 0)
                        grad.at(i, j, k, l) = scale;
                    else if (diff < 0)
                        grad.at(i, j, k, l) = -scale;
                    else
                        grad.at(i, j, k, l) = 0.0f;
                }
            }
        }
    }
    return grad;
}
