//
// Created by muham on 5.12.2025.
//

#include "CortexMind/net/ActivationFunc/Softmax/softmax.hpp"
#include <CortexMind/framework/Log/log.hpp>
#include <stdexcept>

using namespace cortex::act;
using namespace cortex;

Softmax::Softmax() : outputCache(0, 0, 0) {}

tensor Softmax::forward(const tensor &input) {
    if (input.get_shape().size() != 2) {
        log("Softmax activation function only supports 2D tensors.");
        throw std::invalid_argument("Softmax activation function only supports 2D tensors.");
    }

    const size_t batch = input.get_shape()[0];
    const size_t features = input.get_shape()[1];

    tensor output(batch, features, false);

    for (size_t i = 0; i < batch; ++i) {
        double max_val = -std::numeric_limits<double>::infinity();

        for (size_t j = 0; j < features; ++j) {
            max_val = std::max(max_val, input.get_data()[i * features + j]);
        }

        std::vector<double> exp_values(features);
        double exp_sum = 0.0;

        for (size_t j = 0; j < features; ++j) {
            const double stable_logit = input.get_data()[i * features + j] - max_val;
            const double exp_val = std::exp(stable_logit);
            exp_values[j] = exp_val;
            exp_sum += exp_val;
        }

        if (exp_sum == 0.0) {
            log("Sum of exponential is zero in Softmax forward pass.");
            throw std::runtime_error("Sum of exponential is zero in Softmax forward pass.");
        }

        for (size_t j = 0; j < features; ++j) {
            output.get_data()[i * features + j] = exp_values[j] / exp_sum;
        }
    }
    this->outputCache = output;
    return output;
}

tensor Softmax::backward(const tensor &grad_output) {
    if (grad_output.get_shape() != this->outputCache.get_shape()) {
        log("Gradient output shape does not match cached output shape in Softmax backward pass.");
        throw std::invalid_argument("Gradient output shape does not match cached output shape in Softmax backward pass.");
    }

    const size_t batch_size = this->outputCache.get_shape()[0];
    const size_t features = this->outputCache.get_shape()[1];

    tensor output(batch_size, features, false);

    for (size_t b = 0; b < batch_size; ++b) {
        double sum_y_times_grad = 0.0;
        for (size_t f = 0; f < features; ++f) {
            const size_t idx = b * features + f;
            sum_y_times_grad += this->outputCache.get_data()[idx] * grad_output.get_data()[idx];
        }

        for (size_t f = 0; f < features; ++f) {
            const size_t idx = b * features + f;
            const double y_hat = this->outputCache.get_data()[idx];
            const double grad_out = grad_output.get_data()[idx];
            output.get_data()[idx] = y_hat * (grad_out - sum_y_times_grad);
        }
    }

    return output;
}
