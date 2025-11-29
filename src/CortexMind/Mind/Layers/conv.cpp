//
// Created by muham on 8.11.2025.
//

#include "CortexMind/Mind/NeuralNetwork/conv.hpp"

#include <CortexMind/Utils/MathTools/random.hpp>

using namespace cortex::nn;

Conv2D::Conv2D() : weights(3, 3), biases(1, 1), lastOutput({}, {}), lastInput({}, {}), mind_kernel_(nullptr) {
    this->mind_kernel_ = std::make_unique<tools::MindKernel>(random_seed());
}

Conv2D::~Conv2D() {
    this->mind_kernel_.reset();
}

cortex::tensor Conv2D::forward(const tensor &input) {
    const auto outputVec = this->mind_kernel_->apply(input.data());

    const size h = outputVec.size();
    const size w = outputVec[0].size();

    tensor output(h, w);

    for (size i = 0; i < h; ++i) {
        for (size j = 0; j < w; ++j) {
            output(i, j) = outputVec[i][j] + this->biases(0, 0);
        }
    }

    this->lastOutput = output;
    this->lastInput = input;

    return output;
}

cortex::tensor Conv2D::backward(const tensor &grad_output) {
    return grad_output;
}

void Conv2D::setParams(const tensor &params) {
    this->weights = params;
    this->biases(0, 0) = params(0, 0);
}