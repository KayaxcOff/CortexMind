//
// Created by muham on 19.04.2026.
//

#include "CortexMind/utils/Cast/factory.hpp"

using namespace cortex::_fw::sys;
using namespace cortex::utils;
using namespace cortex;

TensorFactory::TensorFactory(const float32 value) : value(value) {}

tensor TensorFactory::cast(const deviceType d_tye, const boolean requires_grad) const {
    auto output = tensor({1}, d_tye, requires_grad);
    output.fill(value);
    return output;
}