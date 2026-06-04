//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw;

Tensor::Tensor() : m_shape({}), m_require(false) {
    this->storage_ = nullptr;
    this->flow_ = nullptr;
    this->gradient_ = nullptr;
}

Tensor::Tensor(const std::initializer_list<i64> &_shape, const sys::DeviceType _device, const bool _requires_grad) : m_shape(_shape), m_require(_requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape, this->m_shape.ndim), _device);

    if (this->m_require) {
        this->gradient_ = std::make_shared<Tensor>(_shape, _device);
        this->gradient_->zero();
    }
}

Tensor::Tensor(const meta::GradientPacked &packed) : m_shape({}), m_require(packed.has_gradient) {
    this->storage_ = packed.stor;
    this->flow_ = packed.flow;

    if (this->m_require) {
        this->gradient_ = packed.gradient;
    }
}

Tensor::Tensor(const Tensor &other) : m_shape(other.m_shape), m_require(other.m_require) {
    this->storage_ = other.storage_;
    this->flow_ = other.flow_;

    if (this->m_require) {
        this->gradient_ = other.gradient_;
    }
}

Tensor::Tensor(Tensor &&other) noexcept : m_shape(other.m_shape), m_require(other.m_require) {
    this->storage_ = std::move(other.storage_);
    this->flow_ = std::move(other.flow_);

    if (this->m_require) {
        this->gradient_ = std::move(other.gradient_);
    }

    other.storage_ = nullptr;
    other.flow_ = nullptr;
    other.gradient_ = nullptr;
}

Tensor::~Tensor() = default;

f32 *Tensor::get() {
    return this->storage_->data() + this->m_shape.offset;
}

const f32 *Tensor::get() const {
    CXM_ASSERT(this->storage_ == nullptr, "Storage is nullptr");
    return this->storage_->data() + this->m_shape.offset;
}

std::span<const i64> Tensor::shape() const {
    return {this->m_shape.shape.data(), static_cast<size_t>(m_shape.ndim)};
}

bool Tensor::has_grad() const {
    return this->gradient_ != nullptr;
}

bool Tensor::is_require() const {
    return this->m_require;
}

bool Tensor::empty() const {
    return this->storage_->isEmpty();
}