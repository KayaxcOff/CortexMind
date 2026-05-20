//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/element_wise.hpp>
#include <CortexMind/framework/Engine/IX/fill.hpp>
#include <CortexMind/framework/Engine/IX/matrix.hpp>
#include <CortexMind/framework/Engine/IX/random.hpp>
#include <CortexMind/framework/Engine/IX/reduce.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.cuh>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <concepts>
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

Tensor::Tensor(const meta::GradientPacked &packed) : m_shape(packed.shape), m_offset(0), m_requires_grad(packed.has_gradient) {
    this->storage_ = packed.stor;
    this->flow_ = packed.flow;

    this->m_strides = compute_stride(this->m_shape);

    if (this->m_requires_grad) {
        CXM_ASSERT(packed.gradient == nullptr, "Gradient of pack is null");

        this->gradient_ = packed.gradient;
    }
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

    if (this->flow_) {
        this->flow_->backward(*this->gradient_);
    }
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

Tensor Tensor::matmul(const Tensor &other) const {
    CXM_ASSERT(this->ndim() != 2 || other.ndim() != 2,
        "matmul requires 2D tensors, got " +
        std::to_string(this->ndim()) + "D and " +
        std::to_string(other.ndim()) + "D");

    CXM_ASSERT(this->m_shape[1] != other.m_shape[0],
        "matmul shape mismatch: inner dimensions must match, got " +
        std::to_string(this->m_shape[1]) + " and " +
        std::to_string(other.m_shape[0]));

    CXM_ASSERT(this->device() != other.device(),
        "matmul device mismatch");

    CXM_ASSERT(!this->contiguous() || !other.contiguous(),
    "matmul requires contiguous tensors");

    const auto M = static_cast<size_t>(this->m_shape[0]);
    const auto K = static_cast<size_t>(this->m_shape[1]);
    const auto N = static_cast<size_t>(other.m_shape[1]);

    Tensor output({static_cast<i64>(M), static_cast<i64>(N)}, this->device(), this->m_requires_grad || other.m_requires_grad);

    MatrixOp::matmul(
        this->storage_.get(), other.storage_.get(),
        output.storage_.get(),
        M, K, N,
        this->device()
    );

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};
        meta::GradientPacked y {other.storage_, other.flow_, other.gradient_, other.m_shape, other.m_requires_grad};

        output.flow_ = std::make_shared<meta::matmul>(x, y);
    }

    return output;
}

Tensor Tensor::transpose() const {
    return {{this->m_shape[1], this->m_shape[0]}, this->storage_, this->m_requires_grad};
}

Tensor Tensor::permute(const std::vector<i64> &dims) const {
    CXM_ASSERT(dims.size() != this->ndim(),
        "permute: dims size must match tensor ndim");

    std::vector seen(this->ndim(), false);
    for (const i64 d : dims) {
        CXM_ASSERT(d < 0 || d >= static_cast<i64>(this->ndim()), "permute: invalid dim " + std::to_string(d));
        CXM_ASSERT(seen[static_cast<size_t>(d)], "permute: duplicate dim " + std::to_string(d));
        seen[static_cast<size_t>(d)] = true;
    }

    std::vector<i64> new_shape(this->ndim());
    std::vector<i64> new_strides(this->ndim());

    for (size_t i = 0; i < dims.size(); ++i) {
        const auto d = static_cast<size_t>(dims[i]);
        new_shape[i]   = this->m_shape[d];
        new_strides[i] = this->m_strides[d];
    }

    Tensor output(new_shape, new_strides, this->storage_, this->m_requires_grad);

    output.m_offset = this->m_offset;

    if (this->m_requires_grad && this->gradient_ != nullptr) {
        output.gradient_ = this->gradient_;
    }

    output.flow_ = this->flow_;

    return output;
}

Tensor Tensor::reshape(const std::vector<i64> &_new_shape) const {
    CXM_ASSERT(!this->contiguous(), "reshape requires a contiguous tensor");

    Tensor output(_new_shape, this->storage_, this->m_requires_grad);
    output.m_offset = this->m_offset;
    if (this->m_requires_grad) {
        output.gradient_ = this->gradient_;
    }
    output.flow_ = this->flow_;
    return output;
}

Tensor Tensor::log() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::log(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::log>(x);
    }

    return output;
}

Tensor Tensor::exp() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::exp(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::exp>(x);
    }

    return output;
}

Tensor Tensor::pow(const f32 exp) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::pow(this->storage_.get(), exp, output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::pow>(x, exp);
    }

    return output;
}

Tensor Tensor::sqrt() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::sqrt(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::sqrt>(x);
    }

    return output;
}

Tensor Tensor::rsqrt() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::rsqrt(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::rsqrt>(x);
    }

    return output;
}

Tensor Tensor::sin() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::sin(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::sin>(x);
    }

    return output;
}

Tensor Tensor::cos() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::cos(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::cos>(x);
    }

    return output;
}

Tensor Tensor::abs() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::abs(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::abs>(x);
    }

    return output;
}

Tensor Tensor::slice(const i64 dim, const i64 start, const i64 end) const {
    CXM_ASSERT(dim < 0 || dim >= static_cast<i64>(this->ndim()),
        "slice: invalid dim " + std::to_string(dim));
    CXM_ASSERT(start < 0 || start >= this->m_shape[dim],
        "slice: invalid start " + std::to_string(start));
    CXM_ASSERT(end <= start || end > this->m_shape[dim],
        "slice: invalid end " + std::to_string(end));

    const auto d = static_cast<size_t>(dim);

    std::vector<i64> new_shape   = this->m_shape;
    const std::vector<i64> new_strides = this->m_strides;
    new_shape[d] = end - start;

    const i64 new_offset = this->m_offset + start * this->m_strides[d];

    Tensor output(new_shape, new_strides, this->storage_, this->m_requires_grad);
    output.m_offset = new_offset;

    if (this->m_requires_grad && this->gradient_ != nullptr) {
        output.gradient_ = this->gradient_;
    }
    output.flow_ = this->flow_;

    return output;
}

Tensor Tensor::sum() const {
    Tensor output({1}, this->storage_->device(), this->m_requires_grad);

    output.get()[0] = this->sum_all();

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::sum>(x);
    }

    return output;
}

Tensor Tensor::sum(const std::vector<i64> &dims) const {
    std::vector<i64> out_shape = this->m_shape;

    for (const i64 d : dims) {
        out_shape[static_cast<size_t>(d)] = 1;
    }

    Tensor output(out_shape, this->device(), this->m_requires_grad);
    output.zero();

    const size_t total = this->len();
    const size_t ndim  = this->ndim();

    for (size_t i = 0; i < total; ++i) {
        size_t oz = 0, idx = i;
        for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
            const size_t coord = idx % static_cast<size_t>(this->m_shape[d]);
            idx /= static_cast<size_t>(this->m_shape[d]);

            const bool is_reduced = std::ranges::find(dims, static_cast<i64>(d)) != dims.end();

            const size_t out_coord = is_reduced ? 0 : coord;
            oz += out_coord * static_cast<size_t>(compute_stride(out_shape)[static_cast<size_t>(d)]);
        }
        output.get()[oz] += this->get()[i];
    }

    return output;
}

Tensor Tensor::neg() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::neg(this->storage_.get(), output.storage_.get(), this->len());

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::neg>(x);
    }

    return output;
}

Tensor Tensor::sign() const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ElementWise::sign(this->storage_.get(), output.storage_.get(), this->len());

    return output;
}

Tensor Tensor::squeeze(const i64 dim) const {
    CXM_ASSERT(dim < 0 || dim >= static_cast<i64>(this->ndim()),
        "squeeze: invalid dim " + std::to_string(dim));
    CXM_ASSERT(this->m_shape[static_cast<size_t>(dim)] != 1,
        "squeeze: dim " + std::to_string(dim) + " is not size 1");

    std::vector<i64> new_shape, new_strides;
    for (size_t d = 0; d < this->ndim(); ++d) {
        if (static_cast<i64>(d) == dim) continue;
        new_shape.push_back(this->m_shape[d]);
        new_strides.push_back(this->m_strides[d]);
    }

    Tensor output(new_shape, new_strides, this->storage_, this->m_requires_grad);
    output.m_offset = this->m_offset;

    if (this->m_requires_grad && this->gradient_ != nullptr) {
        output.gradient_ = this->gradient_;
    }
    output.flow_ = this->flow_;

    return output;
}

Tensor Tensor::unsqueeze(const i64 dim) const {
    const i64 ndim = static_cast<i64>(this->ndim());
    CXM_ASSERT(dim < 0 || dim > ndim,
        "unsqueeze: invalid dim " + std::to_string(dim));

    std::vector<i64> new_shape, new_strides;
    for (i64 d = 0; d <= ndim; ++d) {
        if (d == dim) {
            new_shape.push_back(1);
            new_strides.push_back(d < ndim ? this->m_strides[static_cast<size_t>(d)] : 1);
        }
        if (d < ndim) {
            new_shape.push_back(this->m_shape[static_cast<size_t>(d)]);
            new_strides.push_back(this->m_strides[static_cast<size_t>(d)]);
        }
    }

    Tensor output(new_shape, new_strides, this->storage_, this->m_requires_grad);
    output.m_offset = this->m_offset;

    if (this->m_requires_grad && this->gradient_ != nullptr) {
        output.gradient_ = this->gradient_;
    }
    output.flow_ = this->flow_;

    return output;
}

Tensor Tensor::addition(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad || other.m_requires_grad);

    MatrixOp::add(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides,
        output.storage_.get(),
        output.m_shape,
        output.m_strides
    );

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};
        meta::GradientPacked y {other.storage_, other.flow_, other.gradient_, other.m_shape, other.m_requires_grad};

        output.flow_ = std::make_shared<meta::add>(x, y);
    }

    return output;
}

Tensor Tensor::subtract(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad || other.m_requires_grad);

    MatrixOp::sub(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides,
        output.storage_.get(),
        output.m_shape,
        output.m_strides
    );

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};
        meta::GradientPacked y {other.storage_, other.flow_, other.gradient_, other.m_shape, other.m_requires_grad};

        output.flow_ = std::make_shared<meta::sub>(x, y);
    }

    return output;
}

Tensor Tensor::multiply(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad || other.m_requires_grad);

    MatrixOp::mul(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides,
        output.storage_.get(),
        output.m_shape,
        output.m_strides
    );

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};
        meta::GradientPacked y {other.storage_, other.flow_, other.gradient_, other.m_shape, other.m_requires_grad};

        output.flow_ = std::make_shared<meta::mul>(x, y);
    }

    return output;
}

Tensor Tensor::divide(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad || other.m_requires_grad);

    MatrixOp::div(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides,
        output.storage_.get(),
        output.m_shape,
        output.m_strides
    );

    if (output.m_requires_grad) {
        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};
        meta::GradientPacked y {other.storage_, other.flow_, other.gradient_, other.m_shape, other.m_requires_grad};

        output.flow_ = std::make_shared<meta::div>(x, y);
    }

    return output;
}

Tensor Tensor::clone() const {
    Tensor output;
    output.m_shape         = this->m_shape;
    output.m_strides       = compute_stride(this->m_shape);
    output.m_requires_grad = this->m_requires_grad;
    output.m_offset        = this->m_offset;
    output.storage_        = std::make_shared<TensorStorage>(this->storage_->clone());
    output.flow_           = this->flow_;

    if (this->m_requires_grad && this->gradient_ != nullptr) {
        output.gradient_ = std::make_shared<Tensor>(this->gradient_->clone());
    }

    return output;
}

Tensor &Tensor::grad() {
    CXM_ASSERT(!this->m_requires_grad, "Tensor::grad() requires grad");
    return *this->gradient_;
}

const Tensor &Tensor::grad() const {
    CXM_ASSERT(!this->m_requires_grad, "Tensor::grad() requires grad");
    return *this->gradient_;
}

Tensor::Tensor(const std::vector<i64> &shape, const std::shared_ptr<TensorStorage> &storage, const bool _requires_grad) : m_shape(shape), m_requires_grad(_requires_grad) {
    this->storage_ = storage;

    this->m_strides = compute_stride(this->m_shape);

    this->m_offset = 0;

    if (this->m_requires_grad) {
        this->gradient_ = std::make_shared<Tensor>(this->m_shape, this->storage_->device());
        this->gradient_->zero();
    }
}

Tensor::Tensor(const std::vector<i64> &shape, const std::vector<i64> &stride, const std::shared_ptr<TensorStorage> &storage, const bool _requires_grad) : m_shape(shape), m_strides(stride), m_offset(0), m_requires_grad(_requires_grad) {
    this->storage_ = storage;

    if (this->m_requires_grad) {
        this->gradient_ = std::make_shared<Tensor>(this->m_shape, this->storage_->device());
        this->gradient_->zero();
    }
}