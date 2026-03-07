//
// Created by muham on 6.03.2026.
//

#include "CortexMind/net/NeuralNetwork/leaky_relu.hpp"
#include <CortexMind/core/Graph/flow_ops.hpp>
#include <string>

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

LeakyReLU::LeakyReLU(const float32 alpha) : Layer(true, "LeakyReLU(" + std::to_string(alpha) + ")"), alpha(alpha) {}

LeakyReLU::~LeakyReLU() = default;

tensor LeakyReLU::forward(tensor &input) {
    this->last_input = input;
    this->last_input.clear_flow();

    const size_t num = this->last_input.numel();
    tensor output(this->last_input.shape(), this->last_input.devices(), this->last_input.requires_grad());

    const float32* px = this->last_input.get();
    float32*       py = output.get();
    for (size_t i = 0; i < num; ++i) py[i] = px[i] > 0.0f ? px[i] : this->alpha * px[i];

    if (this->last_input.requires_grad()) {
        output.set_flow(std::make_shared<meta::leaky_relu>(
            &this->last_input, this->alpha
        ));
    }
    return output;
}

std::vector<ref<tensor> > LeakyReLU::parameters() {
    return {};
}

std::vector<ref<tensor> > LeakyReLU::gradients() {
    return {};
}
