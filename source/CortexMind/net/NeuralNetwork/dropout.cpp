//
// Created by muham on 5.03.2026.
//

#include "CortexMind/net/NeuralNetwork/dropout.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <CortexMind/core/Graph/flow_ops.hpp>
#include <random>
#include <string>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Dropout::Dropout(const float32 p) : Layer(true, "Dropout(" + std::to_string(p) + ")"), p(p) {
    CXM_ASSERT(p >= 0.0f && p < 1.0f, "cortex::nn::Dropout::Dropout()", "p must be in [0, 1).");
}

Dropout::~Dropout() = default;

tensor Dropout::forward(tensor& input) {
    this->last_input = input;

    if (!this->flag) {
        return this->last_input;
    }

    const float32 scale = 1.0f / (1.0f - this->p);
    const auto   n     = static_cast<int64>(this->last_input.numel());

    this->mask = tensor(this->last_input.shape(), false);

    thread_local std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(1.0f - this->p);

    for (int64 i = 0; i < n; ++i)
        this->mask.get()[i] = dist(gen) ? scale : 0.0f;

    tensor output = this->last_input * this->mask;

    if (this->last_input.requires_grad()) {
        output.set_flow(std::make_shared<meta::dropout>(&this->last_input, &this->mask));
    }

    return output;
}

std::vector<ref<tensor> > Dropout::parameters() {
    return {};
}

std::vector<ref<tensor> > Dropout::gradients() {
    return {};
}