//
// Created by muham on 26.02.2026.
//

#include "CortexMind/utils/Cast/factory.hpp"

using namespace cortex::utils;
using namespace cortex::_fw;
using namespace cortex;

TensorFactory::TensorFactory() : value{0} {}

void TensorFactory::set(const float32 _value) {
    this->value = _value;
}

tensor TensorFactory::cast(const bool _requires_grad, const sys::device _dev) const {
    tensor output({1}, _dev, _requires_grad);
    output.fill(this->value);
    return output;
}
