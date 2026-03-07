//
// Created by muham on 1.03.2026.
//

#include "CortexMind/net/NeuralNetwork/sigmoid.hpp"
#include <CortexMind/core/Graph/flow_ops.hpp>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

Sigmoid::Sigmoid() : Layer(true, "Sigmoid") {}

Sigmoid::~Sigmoid() = default;

tensor Sigmoid::forward(tensor &input) {
    this->last_input = input;
    this->last_input.clear_flow();
    this->last_output = this->last_input.sigmoid();

    if (this->last_input.requires_grad()) {
        this->last_output.set_flow(std::make_shared<meta::sigmoid>(
            &this->last_input
        ));
    }
    return this->last_output;
}

std::vector<ref<MindTensor> > Sigmoid::parameters() {
    return {};
}

std::vector<ref<MindTensor> > Sigmoid::gradients() {
    return {};
}
