//
// Created by muham on 9.11.2025.
//

#include "CortexMind/Mind/LossFunc/mae.hpp"

using namespace cortex::loss;

MeanAbsolute::MeanAbsolute() = default;

MeanAbsolute::~MeanAbsolute() = default;

cortex::tensor MeanAbsolute::forward(const tensor &y_true, const tensor &y_pred) {
    const size cols = y_true.get_cols();
    const size rows = y_true.get_rows();

    if (cols != y_pred.get_cols() || rows != y_pred.get_rows()) {
        throw std::invalid_argument("MeanAbsolute Loss: y_true and y_pred must have the same shape.");
    }

    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("MeanAbsolute Loss: tensors cannot be empty.");
    }

    float64 sum_abs = 0;

    for (size i = 0; i < rows; ++i) {
        for (size j = 0; j < cols; ++j) {
            sum_abs += std::abs(y_true(i, j) - y_pred(i, j));
        }
    }

    const float64 avg = sum_abs / static_cast<float64>(rows * cols);

    tensor out(1, 1);
    out(0, 0) = avg;
    return out;
}

cortex::tensor MeanAbsolute::backward(const tensor &y_true, const tensor &y_pred) {
    const size cols = y_true.get_cols();
    const size rows = y_true.get_rows();

    if (cols != y_pred.get_cols() || rows != y_pred.get_rows()) {
        throw std::invalid_argument("MeanAbsolute Loss: y_true and y_pred must have the same shape.");
    }

    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("MeanAbsolute Loss: tensors cannot be empty.");
    }

    tensor gradient(rows, cols);
    const auto n = static_cast<float64>(rows * cols);

    for (size i = 0; i < rows; ++i) {
        for (size j = 0; j < cols; ++j) {
            if (const float64 diff = y_true(i, j) - y_pred(i, j); diff > 0)
                gradient(i, j) = 1.0 / n;
            else if (diff < 0)
                gradient(i, j) = -1.0 / n;
            else
                gradient(i, j) = 0.0;
        }
    }

    return gradient;
}
