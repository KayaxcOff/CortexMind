//
// Created by muham on 26.05.2026.
//

#include "CortexMind/net/NeuralNetwork/dropout.hpp"
#include <string>

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Dropout::Dropout(const float32 rate) : LayerBase("Dropout(" + std::to_string(rate) + ")") {
    this->rate = rate;
}

Dropout::~Dropout() = default;

tensor Dropout::forward(const tensor &input) {
    if (!this->flag()) {
        return input;
    }

    const tensor noise(input.shape(), input.device());
    noise.uniform(0.0f, 1.0f);

    const tensor mask = noise.sub(1.0f - this->rate)
                             .clamp(-1.0f, 0.0f)
                             .neg()
                             .sign();


    const float32 scale = 1.0f / (1.0f - this->rate);

    return input * mask * scale;
}

std::vector<ref<tensor>> Dropout::getParameters() {
    return {};
}

std::vector<ref<tensor>> Dropout::getGradients() {
    return {};
}
