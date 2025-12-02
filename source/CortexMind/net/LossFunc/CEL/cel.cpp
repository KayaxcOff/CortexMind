//
// Created by muham on 2.12.2025.
//

#include "CortexMind/net/LossFunc/CEL/cel.hpp"
#include <CortexMind/framework/Log/log.hpp>
#include <stdexcept>

using namespace cortex::loss;
using namespace cortex;

CrossEntropy::CrossEntropy(const double _epsilon) : epsilon(_epsilon) {}

tensor CrossEntropy::forward(const tensor &predictions, const tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
        log("Predictions and targets must have the same shape for Cross Entropy Loss.");
        throw std::invalid_argument("Shape mismatch between predictions and targets.");
    }

    const size_t total_size = predictions.get_data().size();
    if (total_size == 0) {
        auto output = tensor(1, 1, 1);
        output.fill(0.0);
        return output;
    }

    double loss_sum = 0.0;
    const size_t N = predictions.get_shape()[0] * predictions.get_shape()[1];

    for (size_t n = 0; n < total_size; ++n) {
        double pred = predictions.get_data()[n];

        if (double target = targets.get_data()[n]; target != 0.0) {
            pred = std::max(this->epsilon, target);
            pred = std::min(1.0 - this->epsilon, pred);

            loss_sum += target * std::log(pred);
        }
    }

    const double cce_loss = -loss_sum / static_cast<double>(N);

    tensor output(1, 1, 1, false);
    output(0, 0, 0) = cce_loss;

    return output;
}

tensor CrossEntropy::backward(const tensor &predictions, const tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
        log("Shape mismatch between predictions and targets.");
        throw std::invalid_argument("Shape mismatch between predictions and targets.");
    }

    const size_t total_size = predictions.get_data().size();
    if (total_size == 0) {
        auto output = tensor(1, 1, 1, false);
        output.fill(0.0);
        return output;
    }

    tensor output(predictions.get_shape()[0], predictions.get_shape()[1], predictions.get_shape()[2]);

    const size_t N = predictions.get_shape()[0] * predictions.get_shape()[1];
    const double scale = -1.0 / static_cast<double>(N);

    for (size_t n = 0; n < total_size; ++n) {
        double pred = predictions.get_data()[n];
        const double target = targets.get_data()[n];

        pred = std::max(this->epsilon, pred);
        const double grad_val = target / pred;

        output.get_data()[n] = scale * grad_val;
    }
    return output;
}
