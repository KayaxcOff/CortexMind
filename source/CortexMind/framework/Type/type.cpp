//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Type/type.hpp"
#include <CortexMind/framework/Type/as_string.hpp>

using namespace cortex::_fw;

TensorType::TensorType() {
    this->m_type = DType::Unknown;
}

TensorType::TensorType(const DType type) {
    this->m_type = type;
}

TensorType::TensorType(const TensorType &) = default;

TensorType::TensorType(TensorType &&) noexcept = default;

TensorType::~TensorType() = default;

DType TensorType::type() const noexcept {
    return this->m_type;
}

std::string_view TensorType::ToString() const noexcept {
    return as_string(this->m_type);
}