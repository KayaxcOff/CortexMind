//
// Created by muham on 4.11.2025.
//

#include "CortexMind/Model/Optimizer/SGD/sgd.hpp"

#include <utility>
#include <stdexcept>

using namespace cortex::optim;
using namespace cortex;

StochasticGradient::StochasticGradient(std::vector<math::MindVector*> _params, std::vector<math::MindVector*> _grads, const double _lr) {
    this->params = std::move(_params);
    this->grads = std::move(_grads);
    this->lr = _lr;
}

void StochasticGradient::step() {
    for (size_t i = 0; i < this->params.size(); ++i) {
        auto& param = *this->params[i];
        auto& grad = *this->grads[i];

        if (params.size() != grads.size()) {
            throw std::runtime_error("SGD::step -> params and grads size mismatch");
        }

        for (size_t j = 0; j < this->params[i]->size(); ++j) {
            param[j] -= this->lr * grad[j];
        }
    }
}