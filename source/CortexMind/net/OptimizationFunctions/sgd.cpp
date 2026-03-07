//
// Created by muham on 1.03.2026.
//

#include "CortexMind/net/OptimizationFunctions/sgd.hpp"
#include <string>

using namespace cortex::opt;
using namespace cortex::_fw;
using namespace cortex;

StochasticGradient::StochasticGradient(const float32 _lr) : Optimization(_lr, "SGD(" + std::to_string(_lr) + ")") {}

StochasticGradient::~StochasticGradient() = default;

void StochasticGradient::update() {
    for (size_t i = 0; i < this->params.size(); ++i) {
        this->params[i].get() -= this->lr * this->grads[i].get();
    }
}
