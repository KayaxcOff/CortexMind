//
// Created by muham on 28.02.2026.
//

#include "CortexMind/net/NeuralNetwork/global_avg.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

GlobalAvgPooling::GlobalAvgPooling() : Layer(true, "GlobalAvgPooling") {}

GlobalAvgPooling::~GlobalAvgPooling() = default;

tensor GlobalAvgPooling::forward(tensor &input) {
    this->last_input = input;
    this->last_input.clear_flow();

    int64 batch = this->last_input.shape()[0];
    int64 channel = this->last_input.shape()[1];
    const int64 height = this->last_input.shape()[2];
    const int64 width = this->last_input.shape()[3];

    tensor output({batch, channel}, this->last_input.devices());
    for (int64 i = 0; i < batch; ++i) {
        for (int64 j = 0; j < channel; ++j) {
            float32 sum = 0.0f;
            for (int64 k = 0; k < height; ++k) {
                for (int64 l = 0; l < width; ++l) {
                    sum += this->last_input.at(i, j, k, l);
                }
            }
            output.at(i, j) = sum / static_cast<float32>(height * width);
        }
    }
    return output;
}

std::vector<ref<tensor> > GlobalAvgPooling::parameters() {
    return {};
}

std::vector<ref<tensor> > GlobalAvgPooling::gradients() {
    return {};
}