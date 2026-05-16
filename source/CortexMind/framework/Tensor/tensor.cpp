//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/fill.hpp>
#include <CortexMind/framework/Engine/IX/random.hpp>
#include <CortexMind/framework/Engine/IX/reduce.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.cuh>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>
#include <string>
#include <type_traits>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

Tensor::Tensor() : m_offset(0), m_requires_grad(false) {
    this->storage_ = nullptr;
    this->flow_ = nullptr;
    this->gradient_ = nullptr;
}

Tensor::Tensor(const std::vector<i64> &shape, const DeviceType _device, const bool _requires_grad) : m_shape(shape), m_offset(0), m_requires_grad(_requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape), _device);

    this->m_strides = compute_stride(this->m_shape);

    if (this->m_requires_grad) {
        this->gradient_ = std::make_shared<Tensor>(this->m_shape, this->storage_->device());
        this->gradient_->zero();
    }
}

Tensor::Tensor(const std::initializer_list<i64> shape, const DeviceType _device, const bool _requires_grad) : Tensor(std::vector(shape), _device, _requires_grad) {}

Tensor::Tensor(const std::vector<i64> &shape, const f32 *data, const DeviceType _device, const bool _requires_grad) : m_shape(shape), m_offset(0), m_requires_grad(_requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape), data, _device);

    this->m_strides = compute_stride(this->m_shape);

    if (this->m_requires_grad) {
        this->gradient_ = std::make_shared<Tensor>(this->m_shape, this->storage_->device());
        this->gradient_->zero();
    }
}

Tensor::Tensor(const meta::GradientPacked &packed) : m_shape(packed.shape), m_offset(0), m_requires_grad(true) {
    this->storage_ = packed.stor;

    this->m_strides = compute_stride(this->m_shape);

    this->gradient_ = packed.gradient;
}

Tensor::Tensor(const Tensor &other) : m_shape(other.m_shape), m_offset(other.m_offset), m_requires_grad(other.m_requires_grad) {
    this->storage_ = other.storage_;

    this->flow_ = other.flow_;

    this->m_strides = other.m_strides;

    if (this->m_requires_grad) {
        this->gradient_ = other.gradient_;
    }
}

Tensor::Tensor(Tensor &&other) noexcept {
    this->storage_ = std::move(other.storage_);
    this->flow_ = std::move(other.flow_);

    this->m_shape = std::move(other.m_shape);
    this->m_strides = std::move(other.m_strides);
    this->m_offset = other.m_offset;

    this->m_requires_grad = other.m_requires_grad;

    if (this->m_requires_grad) {
        this->gradient_ = std::move(other.gradient_);
    }

    other.storage_ = nullptr;
    other.flow_ = nullptr;
    other.gradient_ = nullptr;
}

Tensor::~Tensor() = default;

template<typename... Args> requires (std::integral<Args> && ...)
f32 &Tensor::at(Args... args) {
    CXM_ASSERT(this->storage_ == nullptr, "Tensor storage is null");
    CXM_ASSERT(this->storage_->device() != sys::DeviceType::kHOST,
        "at() is only supported on HOST tensors");

    const std::vector<i64> indices = { static_cast<i64>(args)... };

    CXM_ASSERT(indices.size() != this->m_shape.size(),
        "Index dimension mismatch: got " + std::to_string(indices.size()) +
        " expected " + std::to_string(this->m_shape.size()));

    for (size_t d = 0; d < indices.size(); ++d) {
        CXM_ASSERT(indices[d] < 0 || indices[d] >= this->m_shape[d],
            "Index out of bounds at dim " + std::to_string(d) +
            ": got " + std::to_string(indices[d]) +
            " size " + std::to_string(this->m_shape[d]));
    }
    const i64 linear = compute_linear_index(this->m_strides, indices, this->m_offset);
    return this->storage_->data()[linear];
}

template<typename... Args> requires (std::integral<Args> && ...)
const f32 &Tensor::at(Args... args) const {
    CXM_ASSERT(this->storage_ == nullptr, "Tensor storage is null");
    CXM_ASSERT(this->storage_->device() != sys::DeviceType::kHOST,
        "at() is only supported on HOST tensors");

    const std::vector<i64> indices = { static_cast<i64>(args)... };

    CXM_ASSERT(indices.size() != this->m_shape.size(),
        "Index dimension mismatch: got " + std::to_string(indices.size()) +
        " expected " + std::to_string(this->m_shape.size()));

    for (size_t d = 0; d < indices.size(); ++d) {
        CXM_ASSERT(indices[d] < 0 || indices[d] >= this->m_shape[d],
            "Index out of bounds at dim " + std::to_string(d) +
            ": got " + std::to_string(indices[d]) +
            " size " + std::to_string(this->m_shape[d]));
    }
    const i64 linear = compute_linear_index(this->m_strides, indices, this->m_offset);
    return this->storage_->data()[linear];
}

f32 *Tensor::get() {
    return this->storage_->data() + this->m_offset;
}

const f32 *Tensor::get() const {
    return this->storage_->data() + this->m_offset;
}

const std::vector<i64> &Tensor::shape() const {
    return this->m_shape;
}

bool Tensor::has_grad() const {
    return this->gradient_ != nullptr;
}

bool Tensor::empty() const {
    return this->storage_->isEmpty();
}

bool Tensor::contiguous() const {
    return is_contiguous(this->m_strides, this->m_shape);
}

DeviceType Tensor::device() const {
    return this->storage_->device();
}

size_t Tensor::len() const {
    return this->storage_->size();
}

size_t Tensor::ndim() const {
    return this->m_shape.size();
}

f32 Tensor::mean() const {
    return reduce::mean(this->storage_.get(), this->len());
}

f32 Tensor::variance() const {
    return reduce::var(this->storage_.get(), this->len());
}

f32 Tensor::stdv() const {
    return reduce::stdv(this->storage_.get(), this->len());
}

f32 Tensor::max() const {
    return reduce::max(this->storage_.get(), this->len());
}

f32 Tensor::min() const {
    return reduce::min(this->storage_.get(), this->len());
}

f32 Tensor::sum_all() const {
    return reduce::sum(this->storage_.get(), this->len());
}

f32 Tensor::norm1() const {
    return reduce::norm1(this->storage_.get(), this->len());
}

f32 Tensor::norm2() const {
    return reduce::norm2(this->storage_.get(), this->len());
}

void Tensor::fill(const f32 value) {
    #if CXM_IS_CUDA_AVAILABLE
        FillOp::fill(this->storage_.get(), value, this->len());
    #else //#if CXM_IS_CUDA_AVAILABLE
        for (size_t i = 0; i < this->len(); ++i) {
            this->get()[i] = value;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Tensor::zero() {
    #if CXM_IS_CUDA_AVAILABLE
        FillOp::zero(this->storage_.get(), this->len());
    #else //#if CXM_IS_CUDA_AVAILABLE
        for (size_t i = 0; i < this->len(); ++i) {
            this->get()[i] = 0.0f;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Tensor::ones() {
    #if CXM_IS_CUDA_AVAILABLE
        FillOp::ones(this->storage_.get(), this->len());
    #else //#if CXM_IS_CUDA_AVAILABLE
        for (size_t i = 0; i < this->len(); ++i) {
            this->get()[i] = 1.0f;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Tensor::uniform(const f32 min, const f32 max) const {
    RandomOp::uniform(this->storage_.get(), min, max, this->len());
}

void Tensor::backward() const {
    CXM_ASSERT(this->gradient_ == nullptr, "Gradient of tensor is null");

    if (this->len() == 1) {
        this->gradient_->ones();
    }

    this->flow_->backward(*this->gradient_);
}

void Tensor::backward(const Tensor &_grad) const {
    if (this->flow_ == nullptr) {
        return;
    }
    this->flow_->backward(_grad);
}

void Tensor::SetData(const f32 *_data) {
    #if CXM_IS_CUDA_AVAILABLE
        if (this->storage_->device() == DeviceType::kHOST) {
            transform::copy_h2h(this->get(), _data, this->len());
        } else {
            transform::copy_d2d(this->get(), _data, this->len());
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::memcpy(this->get(), _data, this->len() * sizeof(f32));
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void Tensor::SetGrad(const std::shared_ptr<Tensor> &_grad) {
    CXM_ASSERT(!this->m_requires_grad, "Gradient is not active");
    this->gradient_ = _grad;
}

void Tensor::SetGrad(const Tensor &_grad) {
    CXM_ASSERT(!this->m_requires_grad, "Gradient is not active");
    this->gradient_ = std::make_shared<Tensor>(_grad);
}

void Tensor::SetFlow(const std::shared_ptr<meta::GradientFlow> &_flow) {
    this->flow_ = _flow;
}

Tensor Tensor::to(const DeviceType &_device) const {
    this->storage_->SetDevice(_device);

    return *this;
}