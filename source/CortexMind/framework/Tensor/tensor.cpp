//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/TensorInit/init.hpp>
#include <CortexMind/framework/Engine/IX/fill.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.cuh>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

Tensor::Tensor() : m_shape({}), m_require(false) {
    this->storage_ = nullptr;
    this->flow_ = nullptr;
    this->gradient_ = nullptr;
}

Tensor::Tensor(const std::initializer_list<i64> &_shape, const DeviceType _device, const bool _requires_grad) : m_shape(_shape), m_require(_requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape, this->m_shape.ndim), _device);

    if (this->m_require) {
        this->gradient_ = std::make_shared<Tensor>(_shape, _device);
        this->gradient_->zero();
    }
}

Tensor::Tensor(const std::span<const i64> &_shape, const DeviceType _device, const bool _requires_grad) : m_shape(_shape), m_require(_requires_grad) {
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

bool Tensor::is_contiguous() const {
    return _fw::is_contiguous(this->m_shape.stride, this->m_shape.shape, this->m_shape.ndim);
}

DeviceType Tensor::device() const {
    return this->storage_->device();
}

size_t Tensor::len() const {
    return this->storage_->size();
}

size_t Tensor::ndim() const {
    return this->m_shape.ndim;
}

void Tensor::fill(const f32 value) const {
    ix::FillOp::fill(this->storage_.get(), value, this->len());
}

void Tensor::zero() const {
    ix::FillOp::zero(this->storage_.get(), this->len());
}

void Tensor::ones() const {
    ix::FillOp::ones(this->storage_.get(), this->len());
}

void Tensor::randn() const {
    ix::TensorInit::rand(this->storage_.get());
}

void Tensor::require_grad() {
    CXM_WARN(this->m_require == true, "Gradient is already required");
    this->m_require = true;

    if (this->gradient_ == nullptr) {
        this->gradient_ = std::make_shared<Tensor>(this->shape(), this->device());
        this->gradient_->zero();
    }
}

void Tensor::uniform(const f32 min, const f32 max) const {
    ix::TensorInit::uniform(this->storage_.get(), min, max);
}

void Tensor::backward() const {
    CXM_ASSERT(this->flow_ == nullptr, "Gradient Flow is null");
    CXM_ASSERT(this->m_require == false, "Gradient isn't require");

    if (this->len() == 1) {
        this->gradient_->ones();
    }

    this->flow_->backward(*this->gradient_);
}

void Tensor::backward(const Tensor &_grad) const {
    if (this->flow_) {
        this->flow_->backward(_grad);
    }
}

void Tensor::SetData(const f32 *_data) {
    #if CXM_IS_CUDA_AVAILABLE
        if (this->storage_->device() == DeviceType::kHOST) {
            transform::copy_h2h(this->get(), _data, this->len());
        } else {
            transform::upload(this->get(), _data, this->len());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::memcpy(this->get(), _data, this->len() * sizeof(f32));
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Tensor::SetGrad(const std::shared_ptr<Tensor> &_grad) {
    this->gradient_ = _grad;
}

void Tensor::SetGrad(const Tensor &_grad) {
    this->gradient_ = std::make_shared<Tensor>(_grad);
}

void Tensor::SetFlow(const std::shared_ptr<meta::GradientFlow> &_flow) {
    this->flow_ = _flow;
}
