//
// Created by muham on 9.11.2025.
//

#include "CortexMind/Mind/LossFunc/mse.hpp"

using namespace cortex::loss;

MeanSquared::MeanSquared() = default;

MeanSquared::~MeanSquared() = default;

cortex::tensor MeanSquared::forward(const tensor &y_true, const tensor &y_pred) {
    const auto n = y_true.get_cols();
    const auto m = y_pred.get_cols();

    if (n != y_pred.get_cols() || m != y_pred.get_cols()) {
        throw std::invalid_argument("MeanSquared::forward - y_true and y_pred must have the same shape.");
    }

    float64 sum = 0;

    for (size i = 0; i < n; ++i) {
        for (size j = 0; j < m; ++j) {
            const float64 diff = y_true(i, j) - y_pred(i, j);
            sum += diff * diff;
        }
    }

    float64 avg = sum / static_cast<float64>(n * m);

    return {n, m, avg};
}

cortex::tensor MeanSquared::backward(const tensor &y_true, const tensor &y_pred) {
    const auto n = y_true.get_cols();
    const auto m = y_pred.get_cols();

    if (n != y_pred.get_cols() || m != y_pred.get_cols()) {
        throw std::invalid_argument("MeanSquared::forward - y_true and y_pred must have the same shape.");
    }

    tensor grad(n, m);

    const float64 scale = 2.0 / static_cast<float64>(n * m);

    for (size i = 0; i < n; ++i) {
        for (size j = 0; j < m; ++j) {
            grad(i, j) = scale * y_true(i, j) - y_pred(i, j);
        }
    }

    return grad;
}