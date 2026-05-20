//
// Created by muham on 20.05.2026.
//

#include "CortexMind/net/OptimizationFunction/sgd.hpp"
#include <string>

using namespace cortex::_fw;
using namespace cortex::opt;

StochasticGradient::StochasticGradient(const float32 _lr) : OptimizationBase("SGD(" + std::to_string(_lr) + ")", _lr) {}

StochasticGradient::~StochasticGradient() = default;

void StochasticGradient::update() {
    for (const auto& item : this->parameters()) {
        item.get() -= this->lr() * item.get().grad();
    }
}