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
    /*
    for (const auto& item : this->parameters()) {
        item.get() -= this->lr() * item.get().grad();
    }
    */
    for (auto i : this->parameters()) {
        auto& grad = i.get().grad();

        float max_norm = 100.0f;
        float current_norm = grad.norm2();
        if (current_norm > max_norm) {
            grad *= (max_norm / current_norm);
        }

        i.get() -= this->lr() * grad;
    }
}