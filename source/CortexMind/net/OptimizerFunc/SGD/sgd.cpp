//
// Created by muham on 3.12.2025.
//

#include "CortexMind/net/OptimizerFunc/SGD/sgd.hpp"

using namespace cortex::optim;

StochasticGradient::StochasticGradient(const float learning_rate) : Optimizer(learning_rate) {}

void StochasticGradient::step() {
    const double eta = this->learning_rate;

    for (auto&[_weights, _grads] : this->params_list) {
        tensor* weights = _weights;
        const tensor* gradients = _grads;

        if (!weights || !gradients) {
            std::cerr << "SGD has founds weights" << std::endl;
            continue;
        }

        tensor scale_grad = (*gradients) * eta;
        (*weights) = (*weights) - scale_grad;
    }
}
