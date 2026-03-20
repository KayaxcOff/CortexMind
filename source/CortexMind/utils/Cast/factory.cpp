//
// Created by muham on 19.03.2026.
//

#include "CortexMind/utils/Cast/factory.hpp"

using namespace cortex::utils;
using namespace cortex;

TensorFactory::TensorFactory() : value() {}

void TensorFactory::set(const float32 _value) {
    this->value = _value;
}

tensor TensorFactory::cast(const dev d, const bool requires_grad) const {
    auto output = tensor({1}, d, requires_grad);
    output.fill(this->value);
    return output;
}
