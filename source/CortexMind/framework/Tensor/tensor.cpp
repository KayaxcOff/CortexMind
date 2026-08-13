//
// Created by muham on 12.08.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Tools/tensor_meta.hpp>
#include <CortexMind/framework/Type/size.hpp>

using namespace cortex::_fw;

Tensor::Tensor() {
    this->m_type = TensorType();
    this->m_flag = false;
}

Tensor::Tensor(const tlx::vec<std::int64_t, 5>& shape, const DType type, const DeviceType d_type, const bool _requires_grad) {
    this->m_shape = TensorShape(shape);
    this->m_type = TensorType(type);
    this->m_flag = _requires_grad;

    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape()) * sizeOf(this->m_type.type()), d_type);

    if (this->m_flag) {
        this->gradient_ = std::make_shared<Tensor>(shape, type, d_type);
    }
}

Tensor::Tensor(const std::vector<std::int64_t> &shape, DType type, DeviceType d_type, const bool _requires_grad) {
    this->m_shape = TensorShape(shape);
    this->m_type = TensorType(type);
    this->m_flag = _requires_grad;

    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape()) * sizeOf(this->m_type.type()), d_type);

    if (this->m_flag) {
        this->gradient_ = std::make_shared<Tensor>(shape, type, d_type);
    }
}

Tensor::Tensor(const TensorInfo &info) {
    this->m_shape = TensorShape(info._shape);
    this->m_type = TensorType(info._dtype);
    this->m_flag = info._requires_grad;

    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape()) * sizeOf(this->m_type.type()), info._deviceType);

    if (this->m_flag) {
        this->gradient_ = std::make_shared<Tensor>(info._shape, info._dtype, info._deviceType);
    }
}

Tensor::~Tensor() = default;

bool Tensor::requires_grad() const noexcept {
    return this->m_flag;
}

bool Tensor::empty() const noexcept {
    return this->storage_->isEmpty();
}

std::vector<std::int64_t> Tensor::shape() const noexcept {
    return {this->m_shape.shape().begin(), this->m_shape.shape().end()};
}

DType Tensor::dtype() const noexcept {
    return this->m_type.type();
}

DeviceType Tensor::device() const noexcept {
    return this->storage_->device();
}

std::size_t Tensor::len() const noexcept {
    return compute_size(this->m_shape.shape());
}

std::size_t Tensor::ndim() const noexcept {
    return this->m_shape.shape().size();
}

bool Tensor::has_grad() const noexcept {
    return this->gradient_ != nullptr;
}