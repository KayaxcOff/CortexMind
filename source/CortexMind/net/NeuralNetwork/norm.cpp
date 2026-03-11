//
// Created by muham on 11.03.2026.
//

#include "CortexMind/net/NeuralNetwork/norm.hpp"

using namespace cortex::nn;
using namespace cortex::_fw;
using namespace cortex;

LayerNorm::LayerNorm(const int64 dim, const float32 eps) : Layer(true, "LayerNorm"), epsilon(eps), normalized_dim(dim){
    this->gamma = tensor({dim}, true);
    this->beta = tensor({dim}, true);

    this->gamma.ones();
    this->beta.zero();
}

LayerNorm::~LayerNorm() = default;

tensor LayerNorm::forward(tensor &input) {
    this->last_input = input;
    this->last_input.clear_flow();

    tensor mean = input.sum(-1, true) / static_cast<float32>(this->normalized_dim);

    tensor centered = input - mean;

    tensor var = (centered.pow(2)).sum(-1, true) / static_cast<float32>(this->normalized_dim);

    tensor std = (var + this->epsilon).sqrt();

    tensor normalized = centered / std;

    tensor output = normalized * this->gamma + this->beta;

    return output;
}

std::vector<ref<tensor>> LayerNorm::parameters() {
    return {this->gamma, this->beta};
}

std::vector<ref<tensor>> LayerNorm::gradients() {
    return {this->gamma.grad(), this->beta.grad()};
}
