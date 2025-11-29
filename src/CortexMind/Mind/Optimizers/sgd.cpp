//
// Created by muham on 9.11.2025.
//

#include "CortexMind/Mind/OptimizerFunc/sgd.hpp"

using namespace cortex::optim;

void StochasticGradient::step(tensor &weights, const tensor &gradients) {
    if (weights.get_rows() != gradients.get_rows() || weights.get_cols() != gradients.get_cols()) {
        throw std::invalid_argument("Weights and gradients must have the same shape.");
    }

    if (weights.get_rows() == 0 || weights.get_cols() == 0) {
        return;
    }

    for (size i = 0; i < weights.get_rows(); ++i) {
        for (size j = 0; j < weights.get_cols(); ++j) {
            weights(i, j) -= this->learning_rate * gradients(i, j);
        }
    }
}