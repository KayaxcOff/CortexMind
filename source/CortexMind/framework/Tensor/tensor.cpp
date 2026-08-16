//
// Created by muham on 12.08.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Memory/transform.cuh>
#include <CortexMind/framework/Tools/Error/errors.hpp>
#include <CortexMind/framework/Type/size.hpp>
#include <tlx/utility.hpp>

using namespace cortex::_fw;

Tensor::Tensor() {
    this->m_type = TensorType();
    this->m_flag = false;
}

Tensor::Tensor(const std::initializer_list<std::int64_t> shape, DType type, DeviceType d_type, const bool _requires_grad) {
    this->m_shape = TensorShape(shape);
    this->m_type = TensorType(type);
    this->m_flag = _requires_grad;

    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape()) * sizeOf(this->m_type.type()), d_type);

    if (this->m_flag) {
        this->gradient_ = std::make_shared<Tensor>(shape, type, d_type);
    }
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

Tensor::Tensor(const Tensor &other) {
    this->m_flag = other.m_flag;

    this->m_shape = other.m_shape;
    this->m_type = other.m_type;

    this->storage_ = other.storage_;
    this->flow_ = other.flow_;

    if (this->m_flag) {
        this->gradient_ = other.gradient_;
    }
}

Tensor::Tensor(Tensor &&other) noexcept {
    this->m_flag = other.m_flag;

    this->m_shape = tlx::move(other.m_shape);
    this->m_type = other.m_type;

    this->storage_ = tlx::move(other.storage_);
    this->flow_ = tlx::move(other.flow_);

    if (this->m_flag) {
        this->gradient_ = tlx::move(other.gradient_);
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

Tensor &Tensor::to(const DeviceType type) {
    if (this->storage_->device() == type) {
        CXM_WARN(true, "Device is already you want");
        return *this;
    }
    CXM_ASSERT(type == DeviceType::Unknown, "Device cannot to be unknown");

    if (device() == DeviceType::HOST) {
        const auto output = std::make_shared<TensorStorage>(this->storage_->bytes(), DeviceType::CUDA);
        transform::upload(output->raw(), this->storage_->raw(), this->storage_->bytes());
        this->storage_ = tlx::move(output);
    } else if (device() == DeviceType::CUDA) {
        const auto output = std::make_shared<TensorStorage>(this->storage_->bytes(), DeviceType::HOST);
        transform::download(output->raw(), this->storage_->raw(), this->storage_->bytes());
        this->storage_ = tlx::move(output);
    } else {
        CXM_DEVICE_ERROR();
    }

    return *this;
}

Tensor &Tensor::grad() noexcept {
    CXM_ASSERT(this->gradient_ == nullptr, "Gradient is null");
    return *this->gradient_;
}

const Tensor &Tensor::grad() const noexcept {
    CXM_ASSERT(this->gradient_ == nullptr, "Gradient is null");
    return *this->gradient_;
}