//
// Created by muham on 14.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Dropout/dropout.hpp"

using namespace cortex::nn;
using namespace cortex;

Dropout::Dropout(const float _dr) : dropout_rate(_dr) {}

tensor Dropout::forward(const tensor &input) {
    this->mask.allocate(input.batch(), input.channel(), input.height(), input.width());
    this->mask.uniform_rand(0.0f, 1.0f);

    for (int i = 0; i < this->mask.batch(); ++i) {
        for (int j = 0; j < this->mask.channel(); ++j) {
            for (int k = 0; k < this->mask.height(); ++k) {
                for (int l = 0; l < this->mask.width(); ++l) {
                    if (this->mask.at(i, j, k, l) < this->dropout_rate) {
                        this->mask.at(i, j, k, l) = 0.0f;
                    } else {
                        this->mask.at(i, j, k, l) = 1.0f;
                    }
                }
            }
        }
    }
    return (input * this->mask) / (1.0f - this->dropout_rate);
}

tensor Dropout::backward(const tensor &grad_output) {
    return (grad_output * this->mask) / (1.0f - this->dropout_rate);
}

std::string Dropout::config() {
    return "Dropout";
}
