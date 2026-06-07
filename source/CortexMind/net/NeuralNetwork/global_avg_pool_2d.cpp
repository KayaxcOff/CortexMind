//
// Created by muham on 30.05.2026.
//

#include "CortexMind/net/NeuralNetwork/global_avg_pool_2d.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

GlobalAveragePool2D::GlobalAveragePool2D() : LayerBase("GlobalAveragePooling2D") {}

GlobalAveragePool2D::~GlobalAveragePool2D() = default;

tensor GlobalAveragePool2D::forward(const tensor &input) {
    return input.mean({2, 3}, false);
}

std::vector<ref<tensor>> GlobalAveragePool2D::getParameters() {
    return {};
}

std::vector<ref<tensor>> GlobalAveragePool2D::getGradients() {
    return {};
}