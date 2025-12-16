//
// Created by muham on 16.12.2025.
//

#include "CortexMind/net/OptimFunc/SGD/sgd.hpp"

using namespace cortex::net;

void StaticGradient::step() {
    for (auto& [weight, grad] : this->iters) {
        *weight -= (*grad) * static_cast<float>(this->learning_rate);
    }
}