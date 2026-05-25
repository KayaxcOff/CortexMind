//
// Created by muham on 22.05.2026.
//

#include "CortexMind/utility/cast.hpp"
#include <limits>

using namespace cortex::_fw::sys;
using namespace cortex::utils;
using namespace cortex;

TensorFactory::TensorFactory(const DeviceType _dev, const bool requires_grad) : m_value(std::numeric_limits<float32>::quiet_NaN()), m_dev(_dev), m_grad_flag(requires_grad) {}

void TensorFactory::Set(const float32 value) {
    this->m_value = value;
}

tensor TensorFactory::cast() const {
    tensor output({1}, this->m_dev, this->m_grad_flag);
    output.fill(this->m_value);
    return output;
}
