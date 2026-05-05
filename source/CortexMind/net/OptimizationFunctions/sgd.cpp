//
// Created by muham on 5.05.2026.
//

#include "CortexMind/net/OptimizationFunctions/sgd.hpp"

using namespace cortex::opt;
using namespace cortex::_fw;

StochasticGradient::StochasticGradient(const float32 _lr) : OptimizationBase("SGD", _lr) {}

StochasticGradient::~StochasticGradient() = default;

void StochasticGradient::update() {
    for (auto item : this->params) {
        item.get() -= item.get().grad() * this->learning_rate;
    }
}
