//
// Created by muham on 4.12.2025.
//

#include "CortexMind/net/NeuralNetwork/MaxPooling/max_pool.hpp"
#include <CortexMind/framework/Log/log.hpp>
#include <stdexcept>

using namespace cortex::nn;
using namespace cortex;

MaxPooling::MaxPooling(const size_t _kernelSize, const size_t _stride) : kernel_size(_kernelSize), stride(_stride), inputCache(0, 0, 0), idxCache(0, 0, 0), in_height(0), in_width(0) {
    if (this->kernel_size < 1 || this->stride < 1) {
        log("Kernel size and stride must be greater than 0.");
        throw std::invalid_argument("Kernel size and stride must be greater than 0.");
    }
}

MaxPooling::~MaxPooling() = default;

tensor MaxPooling::forward(tensor &input) {
    if (input.get_shape().size() != 3) {
        log("Input tensor must be 3-dimensional (channels, height, width).");
        throw std::invalid_argument("Input tensor must be 3-dimensional (channels, height, width).");
    }

    const size_t batch_size = input.get_shape()[0];
    const size_t channels = input.get_shape()[1];

    const size_t flat_size_per_channel = input.get_data().size() / (batch_size * channels);
    const auto input_width = static_cast<size_t>(std::sqrt(flat_size_per_channel));
    const size_t input_height = input_width;

    if (input_width * input_height * channels * batch_size != input.get_data().size()) {
        log("Input tensor dimensions are inconsistent.");
        throw std::invalid_argument("Input tensor dimensions are inconsistent.");
    }

    this->in_height = input_height;
    this->in_width = input_width;
    this->inputCache = input;

    const size_t output_height = (input_height - this->kernel_size) / this->stride + 1;
    const size_t output_width = (input_width - this->kernel_size) / this->stride + 1;

    tensor output(batch_size, channels, output_height * output_width, false);

    this->idxCache = tensor(batch_size, channels, output_height * output_width, false);

    size_t outIdx = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < channels; ++j) {
            const size_t idx = i * channels * input_height * input_width + j * input_height * input_width;

            for (size_t k = 0; k < output_height; ++k) {
                for (size_t n = 0; n < output_width; ++n) {
                    const size_t h_start = k * this->stride;
                    const size_t w_start = n * this->stride;

                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t max_pos_in_input = 0;

                    for (size_t h = 0; h < kernel_size; ++h) {
                        for (size_t w = 0; w < kernel_size; ++w) {

                            const size_t h_in = h_start + h;
                            const size_t w_in = w_start + w;

                            const size_t current_flat_idx = idx + h_in * input_width + w_in;

                            if (const double current_val = input.get_data()[current_flat_idx]; current_val > max_val) {
                                max_val = current_val;
                                max_pos_in_input = current_flat_idx;
                            }
                        }
                    }

                    output.get_data()[outIdx] = max_val;
                    this->idxCache.get_data()[outIdx] = static_cast<double>(max_pos_in_input);
                    outIdx++;
                }
            }
        }
    }
    return output;
}

tensor MaxPooling::backward(tensor &grad_output) {
    if (grad_output.get_data().size() != this->idxCache.get_data().size()) {
        log("Gradient output tensor size does not match the cached indices size.");
        throw std::invalid_argument("Gradient output tensor size does not match the cached indices size.");
    }

    tensor output(this->inputCache.get_shape()[0], this->inputCache.get_shape()[1], this->inputCache.get_shape()[2], false);

    for (size_t i = 0; i < grad_output.get_data().size(); ++i) {
        const auto max_pos_in_input = static_cast<size_t>(this->inputCache.get_data()[i]);
        const double grad_val = grad_output.get_data()[i];
        output.get_data()[max_pos_in_input] += grad_val;
    }

    return output;
}

std::string MaxPooling::config() {
    return "MaxPooling";
}

std::vector<tensor*> MaxPooling::getGradients() {
    return {};
}

std::vector<tensor*> MaxPooling::getParameters() {
    return {};
}