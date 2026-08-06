//
// Created by muham on 6.08.2026.
//

#include "CortexMind/framework/Shape/shape.hpp"
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw;

TensorShape::TensorShape() = default;

TensorShape::TensorShape(const std::initializer_list<std::int64_t> shape) {
    this->m_shape = shape;
    this->m_stride = compute_stride(this->m_shape);
    this->m_offset = 0;
}

TensorShape::TensorShape(const std::vector<std::int64_t> &shape) {
    this->m_shape = tlx::vec<std::int64_t, CXM_MAX_DIMS>(shape);
    this->m_stride = compute_stride(this->m_shape);
    this->m_offset = 0;
}

TensorShape::TensorShape(const TensorShape &) = default;

TensorShape::TensorShape(TensorShape &&) noexcept = default;

TensorShape::~TensorShape() = default;

void TensorShape::Set(const tlx::vec<std::int64_t, 5> &shape) {
    this->m_shape = shape;
    this->m_stride = compute_stride(this->m_shape);
}

tlx::vec<std::int64_t, CXM_MAX_DIMS> &TensorShape::shape() {
    return this->m_shape;
}

const tlx::vec<std::int64_t, CXM_MAX_DIMS> &TensorShape::shape() const {
    return this->m_shape;
}

tlx::vec<std::int64_t, CXM_MAX_DIMS> &TensorShape::stride() {
    return this->m_stride;
}

const tlx::vec<std::int64_t, CXM_MAX_DIMS> &TensorShape::stride() const {
    return this->m_stride;
}

std::int64_t TensorShape::offset() const noexcept {
    return this->m_offset;
}

std::size_t TensorShape::rank() const noexcept {
    return this->m_shape.size();
}