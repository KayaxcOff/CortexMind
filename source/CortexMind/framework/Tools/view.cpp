//
// Created by muham on 18.08.2026.
//

#include "CortexMind/framework/Tools/view.hpp"

using namespace cortex::_fw;

TensorView::TensorView(std::byte *data, const DType type, const DeviceType device, const std::size_t size) {
    this->m_data = data;
    this->m_type = type;
    this->m_device = device;
    this->m_size = size;
}

TensorView::~TensorView() = default;

std::byte *TensorView::data() noexcept {
    return this->m_data;
}

const std::byte *TensorView::data() const noexcept {
    return this->m_data;
}

DType TensorView::dtype() const noexcept {
    return this->m_type;
}

DeviceType TensorView::device() const noexcept {
    return this->m_device;
}

std::size_t TensorView::size() const noexcept {
    return this->m_size;
}