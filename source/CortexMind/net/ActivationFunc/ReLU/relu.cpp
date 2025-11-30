//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/ActivationFunc/ReLU/relu.hpp"
#include <CortexMind/utils/MathTools/pch.hpp>

using namespace cortex::act;
using namespace cortex;

ReLU::ReLU() : cached_relu(0, 0, 0) {}

tensor ReLU::forward(const tensor &input) {
    this->cached_relu = input;
    return relu(this->cached_relu);
}

tensor ReLU::backward(const tensor &grad_output) {
    const tensor deriv = relu_prime(this->cached_relu);
    return deriv * grad_output;
}