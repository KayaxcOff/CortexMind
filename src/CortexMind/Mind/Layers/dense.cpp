//
// Created by muham on 8.11.2025.
//

#include "CortexMind/Mind/NeuralNetwork/dense.hpp"

#include <CortexMind/Utils/MathTools/random.hpp>

using namespace cortex::nn;

Dense::Dense(const size in, const size out) : weights(in, std::vector<float32>(out, 0.0f)), biases(out, 0.0f), gradWeights(in, std::vector<float32>(out, 0.0f)), gradBiases(out, 0.0f), outputGrad(), lastInput() {
    this->weights = {random_weights(in, out)};
}

Dense::~Dense() = default;

cortex::tensor Dense::forward(const tensor &input) {
    if (input.get_cols() <= 0 || input.get_rows() <= 0) {
        throw std::invalid_argument("Input tensor is empty.");
        return tensor{{}, {}};
    }

    const size batch = input.get_cols();
    const size in = input[0].size();
    const size out = this->biases.size();

    if (in != this->weights.size() || out != this->weights[0].size()) {
        throw std::invalid_argument("Input tensor size does not match layer configuration.");
        return tensor{{}, {}};
    }

    cortex::tensor output(batch, std::vector<float32>(out_features, 0.0f));
    this->lastInput = input;

    for (size i=0; i < batch; ++i) {
        for (size j=0; j < out; ++j) {
            float32 sum = this->biases[j];
            for (size k=0; k < in; ++k) {
                sum += input[k][i] * this->weights[k][j];
            }
            output[j][i] = sum;
        }
    }
    return output;
}

cortex::tensor Dense::backward(const tensor &grad_output) {
    const size batch = grad_output.get_cols();
    const size in = this->weights.size();
    const size out = this->weights[0].size();

    this->outputGrad = grad_output;
    cortex::tensor grad_input(in, std::vector<float32>(batch, 0.0f));

    for (size i=0; i < batch; ++i) {
        for (size j=0; j < in; ++j) {
            float32 sum = 0.0f;
            for (size k=0; k < out; ++k) {
                sum += grad_output[k][i] * this->weights[j][k];
                this->gradWeights[j][k] += this->lastInput[j][i] * grad_output[k][i];
            }
            grad_input[j][i] = sum;
        }
    }

    for (size k=0; k < out; ++k) {
        for (size i=0; i < batch; ++i) {
            this->gradBiases[k] += grad_output[k][i];
        }
    }

    return grad_input;
}

cortex::tensor Dense::getParams() const {
    return this->lastInput;
}

cortex::tensor Dense::getGrads() const {
    return this->outputGrad;
}