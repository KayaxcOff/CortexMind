//
// Created by muham on 14.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Embedding/embedding.hpp"
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>

using namespace cortex::nn;
using namespace cortex;

Embedding::Embedding(const int _input_dim, const int _vocab_size) : input_dim(_input_dim), vocab_size(_vocab_size) {
    this->weights.allocate(1, 1, this->vocab_size, this->input_dim);
    this->weights.uniform_rand();
}

Embedding::~Embedding() = default;

tensor Embedding::forward(const tensor &input) {
    this->input_cache = input;

    tensor output(input.batch(), input.channel(), input.height(), this->input_dim);

    for (int i = 0; i < input.batch(); ++i) {
        for (int j = 0; j < input.channel(); ++j) {
            for (int k = 0; k < input.height(); ++k) {
                const int idx = static_cast<int>(input.at(i, j, k, 0));
                if (idx < 0 || idx >= this->vocab_size) CXM_ASSERT(true, "Index out of bounds in Embedding layer forward pass");
                for (int l = 0; l < this->input_dim; ++l) {
                    output.at(i, j, k, l) = this->weights.at(0, 0, idx, l);
                }
            }
        }
    }
    return output;
}

tensor Embedding::backward(const tensor &grad_output) {
    this->grad_weights.allocate(1, 1, this->vocab_size, this->input_dim);

    for (int i = 0; i < this->input_cache.batch(); ++i) {
        for (int j = 0; j < this->input_cache.channel(); ++j) {
            for (int k = 0; k < this->input_cache.height(); ++k) {
                const int idx = static_cast<int>(this->input_cache.at(i, j, k, 0));
                if (idx < 0 || idx >= this->vocab_size) CXM_ASSERT(true, "Index out of bounds in Embedding layer backward pass");
                for (int l = 0; l < this->input_dim; ++l) {
                    this->grad_weights.at(0, 0, idx, l) += grad_output.at(i, j, k, l);
                }
            }
        }
    }
    return tensor();
}

std::string Embedding::config() {
    return "Embedding";
}

std::array<tensor *, 2> Embedding::parameters() {
    return {&this->weights, &this->biases};
}

std::array<tensor *, 2> Embedding::gradients() {
    return {&this->grad_weights, &this->grad_biases};
}