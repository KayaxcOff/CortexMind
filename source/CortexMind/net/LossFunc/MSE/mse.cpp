//
// Created by muham on 19.12.2025.
//

#include "CortexMind/net/LossFunc/MSE/mse.hpp"

using namespace cortex::net;
using namespace cortex;

MeanSquared::MeanSquared() = default;

MeanSquared::~MeanSquared() = default;

tensor MeanSquared::forward(const tensor &predictions, const tensor &targets) const {

    tensor output(1, 1, 1, 1);
    tensor diff = targets - predictions;

    for (int i = 0; i < targets.batch(); ++i) {
        float sum = 0.0f;

        for (int j = 0; j < targets.channel(); ++j) {
            for (int k = 0; k < targets.height(); ++k) {
                for (int l = 0; l < targets.width(); ++l) {
                    const float val = diff.at(i, j, k, l);
                    sum += val * val;
                }
            }
        }

        const int num_elements = targets.channel() * targets.height() * targets.width();
        output.at(i, 0, 0, 0) = sum / static_cast<float>(num_elements);
    }

    return output;
}

tensor MeanSquared::backward(const tensor &predictions, const tensor &targets) const {
    tensor grad(predictions.batch(), targets.channel(), targets.height(), targets.width());

    for (int i = 0; i < targets.batch(); ++i) {
        const auto num = static_cast<float>(targets.channel() * targets.height() * targets.width());
        const float scale = 2.0f / static_cast<float>(num);
        for (int j = 0; j < targets.channel(); ++j) {
            for (int k = 0; k < targets.height(); ++k) {
                for (int l = 0; l < targets.width(); ++l) {
                    grad.at(i, j, k, l) = scale * predictions.at(i, j, k, l) - targets.at(i, j, k, l);
                }
            }
        }
    }
    return grad;
}
