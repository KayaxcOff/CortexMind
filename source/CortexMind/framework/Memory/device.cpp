//
// Created by muham on 3.08.2026.
//

#include "device.hpp"
#include <CortexMind/framework/Memory/as_string.hpp>

using namespace cortex::_fw;

TensorDevice::TensorDevice(const DeviceType type) {
    this->m_type = type;
}

TensorDevice::TensorDevice(const TensorDevice &) = default;

TensorDevice::TensorDevice(TensorDevice &&) noexcept = default;

TensorDevice::~TensorDevice() = default;

DeviceType TensorDevice::type() const noexcept {
    return this->m_type;
}

std::string_view TensorDevice::ToString() const noexcept {
    return as_string(this->m_type);
}