//
// Created by muham on 2.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Dropout/dropout.hpp"

using namespace cortex::nn;
using namespace cortex;

Dropout::Dropout(const double _p) : p(_p), mask(0, 0, 0), is_training(true), generator(std::random_device{}()), distribution(0.0, 1.0) {
    if (this->p < 0.0 || this->p >= 1.0) {
        throw std::invalid_argument("Dropout probability p must be in the range [0, 1).");
    }

    this->scale = 1.0 / (1.0 - this->p);

    this->distribution = std::uniform_real_distribution(0.0, 1.0);
}

void Dropout::train() {
    this->is_training = true;
}

void Dropout::eval() {
    this->is_training = false;
}

tensor Dropout::forward(tensor &input) {
    if (!this->is_training) {
        return input;
    }

    const size_t total_size = input.get_data().size();

    this->mask = tensor(input.get_shape()[0], input.get_shape()[1], input.get_shape()[2], false);

    auto output = tensor(input.get_shape()[0], input.get_shape()[1], input.get_shape()[2], false);

    for (size_t i = 0; i < total_size; i++) {
        if (const double rand_val = this->distribution(this->generator); rand_val < this->p) {
            this->mask.get_data()[i] = 0.0;
            output.get_data()[i] = 0.0;
        } else {
            this->mask.get_data()[i] = this->scale;
            output.get_data()[i] = input.get_data()[i] * this->scale;
        }
    }
    return output;
}

tensor Dropout::backward(tensor &grad_output) {
    if (!this->is_training) {
        return grad_output;
    }
    tensor grad_input(grad_output.get_shape()[0], grad_output.get_shape()[1], grad_output.get_shape()[2], false);
    const size_t total_size = grad_output.get_data().size();

    for (size_t i = 0; i < total_size; ++i) {
        grad_input.get_data()[i] = grad_output.get_data()[i] * this->mask.get_data()[i];
    }

    return grad_input;
}

std::string Dropout::config() {
    return "Dropout";
}