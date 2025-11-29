//
// Created by muham on 8.11.2025.
//

#include "CortexMind/Mind/NeuralNetwork/dense.hpp"

#include <CortexMind/Utils/MathTools/random.hpp>

using namespace cortex::nn;

Dense::Dense(const size in, const size out) : gradBiases(1, out), outputGrad(out, 1), lastInput(0, 0) {
    this->weights.emplace_back(in, out);
    this->gradWeights.emplace_back(in, out);

    this->biases.emplace_back(1, out);

    auto& w = this->weights[0];
    auto& b = this->biases[0];

    for (size i = 0; i < in; i++) {
        for (size j = 0; j < out; j++) {
            w(i, j) = random_weight(0.1f, 1.0f);
        }
    }

    for (size i = 0; i < out; i++) {
        b(0, i) = random_weight(0.1f, 1.0f);
    }
}

Dense::~Dense() = default;

cortex::tensor Dense::forward(const tensor &input) {
    if (input.get_cols() == 0 || input.get_rows() == 0) {
        throw std::runtime_error("Input tensor has invalid dimensions.");
    }

    const size in = this->weights[0].get_rows();
    const size out = this->weights[0].get_cols();
    const size batch = input.get_cols();

    if (input.get_rows() != in) {
        throw std::runtime_error("Input tensor has incompatible dimensions.");
    }

    tensor output(out, batch);
    this->lastInput = input;

    for (size i = 0; i <out ; ++i) {
        for (size j = 0; j < batch; ++j) {
            float64 sum = this->biases[0](0, i);
            for (size k = 0; k < in; ++k) {
                sum += input(k, j) * this->weights[0](k, i);
            }
            output(i, j) = sum;
        }
    }
    return output;
}

cortex::tensor Dense::backward(const tensor &grad_output) {
    const size in = this->weights[0].get_rows();
    const size out = this->weights[0].get_cols();
    const size batch = grad_output.get_cols();

    if (grad_output.get_rows() != out) {
        throw std::runtime_error("grad_output has incompatible dimensions.");
    }

    if (this->lastInput.get_cols() != batch) {
        throw std::runtime_error("lastInput batch size doesn't match grad_output.");
    }

    this->outputGrad = grad_output;

    tensor gradOut(in, batch);

    for (size k = 0; k < in; ++k) {
        for (size j = 0; j < out; ++j) {
            this->gradWeights[0](k, j) = 0.0;
        }
    }

    for (size i = 0; i < out; ++i)
        this->gradBiases(0, i) = 0.0;

    for (size i = 0; i < batch; ++i) {
        for (size j = 0; j < out; ++j) {
            const float64 go = grad_output(j, i);
            for (size k = 0; k < in; ++k) {
                gradOut(k, i) += this->weights[0](k, j) * go;
                this->gradWeights[0](k, j) += this->lastInput(k, i) * go;
            }
            this->gradBiases(0, j) += go;
        }
    }

    return gradOut;
}

void Dense::setParams(const tensor &params) {
    const size in = this->weights[0].get_rows();
    const size out = this->weights[0].get_cols();

    for (size i = 0; i < in; ++i) {
        for (size j = 0; j < out; ++j) {
            this->weights[0](i, j) = params(i, j);
        }
    }

    for (size i = 0; i < out; ++i) {
        this->biases[0](0, i) = params(i, 0);
    }
}

cortex::tensor Dense::getParams() const {
    const size in = weights[0].get_rows();
    const size out = weights[0].get_cols();

    tensor params(in + 1, out);

    for (size i = 0; i < in; i++)
        for (size j = 0; j < out; j++)
            params(i, j) = weights[0](i, j);

    for (size j = 0; j < out; j++)
        params(in, j) = biases[0](0, j);

    return params;
}

cortex::tensor Dense::getGrads() const {
    const size in = gradWeights[0].get_rows();
    const size out = gradWeights[0].get_cols();

    tensor grads(in + 1, out);

    for (size i = 0; i < in; i++)
        for (size j = 0; j < out; j++)
            grads(i, j) = gradWeights[0](i, j);

    for (size j = 0; j < out; j++)
        grads(in, j) = gradBiases(0, j);

    return grads;
}

std::string Dense::get_config() const {
    return "Dense Layer";
}