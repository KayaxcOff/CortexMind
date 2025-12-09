//
// Created by muham on 6.12.2025.
//

#include "CortexMind/net/NeuralNetwork/Dense/dense.hpp"

using namespace cortex::_fw;
using namespace cortex::nn;
using namespace cortex;

Dense::Dense(const size_t input_size, const size_t output_size, std::unique_ptr<ActivationFunc> activation) : Layer(std::move(activation)), in_size(input_size), out_size(output_size) {
    this->weights = tensor(1, static_cast<int>(this->in_size), 1, static_cast<int>(this->out_size));
    this->biases = tensor(1, 1, 1, static_cast<int>(this->out_size));

    this->grad_weights = tensor(1, static_cast<int>(this->in_size), 1, static_cast<int>(this->out_size), 0.0f);
    this->grad_biases = tensor(1, 1, 1, static_cast<int>(this->out_size), 0.0f);

    const float limit = std::sqrt(1.0f / static_cast<float>(this->in_size));
    this->weights.uniform_rand(-limit, limit);
    this->biases.zero();
}

tensor Dense::forward(const tensor &input) {
    this->input_cache = input;

    const tensor x = input.flatten();
    tensor out = x.matmul(this->weights) + this->biases;

    if (this->activ_fn_) out = this->activ_fn_->forward(out);

    return out;
}

tensor Dense::backward(const tensor &grad_output) {
    tensor grad = grad_output;

    if (this->activ_fn_) grad = this->activ_fn_->backward(grad);

    const tensor x = this->input_cache.flatten();
    const tensor w = x.transpose().matmul(grad);

    this->grad_weights += w;

    tensor b(1, 1, 1, static_cast<int>(this->out_size), 0.0f);

    const int B = grad.shapeIdx(0);
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < static_cast<int>(this->out_size); ++j) {
            b.at(0, 0, 0, j) += grad.at(i, j, 0, 0);
        }
    }
    this->grad_biases += b;

    tensor grad_input = grad.matmul(this->weights.transpose());
    return grad_input;
}

std::string Dense::config() const {
    return "Dense";
}

std::vector<std::reference_wrapper<tensor>> Dense::gradients() {
    return {std::ref(this->grad_weights), std::ref(this->grad_biases)};
}

std::vector<std::reference_wrapper<tensor>> Dense::parameters() {
    return {std::ref(this->weights), std::ref(this->biases)};
}