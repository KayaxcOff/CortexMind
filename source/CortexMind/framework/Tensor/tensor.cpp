//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/TensorInit/init.hpp>
#include <CortexMind/framework/Engine/IX/TensorOp/op.hpp>
#include <CortexMind/framework/Engine/IX/TensorReduce/reduce.hpp>
#include <CortexMind/framework/Engine/IX/TensorWise/wise.hpp>
#include <CortexMind/framework/Engine/IX/scalar.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.cuh>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>
#include <bitset>
#include <utility>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;
#include <iostream>
Tensor::Tensor()  {
    std::cout << "Tensor::Tensor()" << std::endl;
    this->m_shape = TensorShape({});
    this->m_require = false;
    this->storage_ = nullptr;
    this->flow_ = nullptr;
    this->gradient_ = nullptr;
}

Tensor::Tensor(const std::initializer_list<i64> &_shape, const DeviceType _device, const bool _requires_grad) : m_shape(_shape), m_require(_requires_grad) {
    std::cout << "Tensor" << std::endl;
    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape), _device);

    if (this->m_require) {
        this->gradient_ = std::make_shared<Tensor>(_shape, _device);
        this->gradient_->zero();
    }
}

Tensor::Tensor(const std::vector<i64> &_shape, const DeviceType _device, const bool _requires_grad) : m_shape(_shape), m_require(_requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape), _device);

    if (this->m_require) {
        this->gradient_ = std::make_shared<Tensor>(_shape, _device);
        this->gradient_->zero();
    }
}

Tensor::Tensor(const TensorShape &_shape, DeviceType _device, const bool _requires_grad) : m_shape(_shape), m_require(_requires_grad) {
    this->storage_ = std::make_shared<TensorStorage>(compute_size(this->m_shape.shape), _device);

    if (this->m_require) {
        this->gradient_ = std::make_shared<Tensor>(_shape, _device);
        this->gradient_->zero();
    }
}

Tensor::Tensor(const meta::GradientPacked &packed) : m_shape(packed.shape), m_require(packed.has_gradient) {
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

Tensor::Tensor(Tensor &&other) noexcept : m_shape(std::move(other.m_shape)), m_require(other.m_require) {
    this->storage_ = std::move(other.storage_);
    this->flow_ = std::move(other.flow_);

    if (this->m_require) {
        this->gradient_ = std::move(other.gradient_);
    }
}

Tensor::~Tensor() = default;

f32 *Tensor::get() {
    return this->storage_->data() + this->m_shape.offset;
}

const f32 *Tensor::get() const {
    CXM_ASSERT(this->storage_ == nullptr, "Storage is nullptr");
    return this->storage_->data() + this->m_shape.offset;
}

const std::vector<i64> &Tensor::shape() const {
    return this->m_shape.shape;
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
    return _fw::is_contiguous(this->m_shape.shape, this->m_shape.stride);
}

DeviceType Tensor::device() const {
    return this->storage_->device();
}

size_t Tensor::len() const {
    return this->storage_->size();
}

size_t Tensor::ndim() const {
    return this->m_shape.shape.size();
}

void Tensor::fill(const f32 value) const {
    ix::TensorInit::fill(this->storage_.get(), value);
}

void Tensor::zero() const {
    ix::TensorInit::fill(this->storage_.get(), 0);
}

void Tensor::ones() const {
    ix::TensorInit::fill(this->storage_.get(), 1);
}

void Tensor::randn() const {
    ix::TensorInit::rand(this->storage_.get());
}

void Tensor::require() {
    CXM_WARN(this->m_require == true, "Gradient already is required");
    this->m_require = true;

    if (this->gradient_ == nullptr) {
        this->gradient_ = std::make_shared<Tensor>(this->shape(), this->device());
        this->gradient_->zero();
    }
}

void Tensor::unrequire() {
    CXM_WARN(this->m_require == false, "Gradient already is not required");
    this->m_require = false;
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

Tensor Tensor::mean() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::mean(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::mean>(this->pack());
    }

    return output;
}

Tensor Tensor::mean(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::mean(this->storage_.get(), output.storage_.get(), outer, dim, inner);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::mean_dim>(this->pack(), dims, keep_dim);
    }
    return output;
}

Tensor Tensor::variance() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::var(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::variance>(this->pack());
    }

    return output;
}

Tensor Tensor::variance(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::var(this->storage_.get(), output.storage_.get(), outer, dim, inner);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::variance_dim>(this->pack(), dims, keep_dim);
    }

    return output;
}

Tensor Tensor::stdv() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::stdv(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::stdv>(this->pack(), output.pack());
    }

    return output;
}

Tensor Tensor::stdv(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::stdv(this->storage_.get(), output.storage_.get(), outer, dim, inner);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::stdv_dim>(this->pack(), output.pack(), dims, keep_dim);
    }

    return output;
}

Tensor Tensor::norm1() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::norm1(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::norm1>(this->pack());
    }

    return output;
}

Tensor Tensor::norm1(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::norm1(this->storage_.get(), output.storage_.get(), outer, dim, inner);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::norm1_dim>(this->pack(), dims, keep_dim);
    }

    return output;
}

Tensor Tensor::norm2() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::norm2(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::norm2>(this->pack(), output.pack());
    }

    return output;
}

Tensor Tensor::norm2(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::norm2(this->storage_.get(), output.storage_.get(), outer, dim, inner);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::norm2_dim>(this->pack(), output.pack(), dims, keep_dim);
    }

    return output;
}

Tensor Tensor::max() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::max(this->storage_.get(), output.storage_.get());
    return output;
}

Tensor Tensor::max(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::max(this->storage_.get(), output.storage_.get(), outer, dim, inner);
    return output;
}

Tensor Tensor::min() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::min(this->storage_.get(), output.storage_.get());
    return output;
}

Tensor Tensor::min(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::min(this->storage_.get(), output.storage_.get(), outer, dim, inner);
    return output;
}

Tensor Tensor::sum() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::sum(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sum>(this->pack());
    }

    return output;
}

Tensor Tensor::sum(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::sum(this->storage_.get(), output.storage_.get(), outer, dim, inner);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sum_dim>(this->pack(), dims, keep_dim);
    }

    return output;
}

Tensor Tensor::argmax() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::argmax(this->storage_.get(), output.m_shape.shape.data());
    return output;
}

Tensor Tensor::argmax(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::argmax(this->storage_.get(), output.m_shape.shape.data(), outer, dim, inner);
    return output;
}

Tensor Tensor::argmin() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::argmin(this->storage_.get(), output.m_shape.shape.data());
    return output;
}

Tensor Tensor::argmin(const std::initializer_list<i64> dims, const bool keep_dim) const {
    size_t outer, dim, inner;
    this->reduce_sizes(dims, outer, dim, inner);

    const std::vector<i64> new_shape = compute_reduced_shape(this->m_shape, dims, keep_dim);

    Tensor output(new_shape, this->device(), this->m_require);

    ix::TensorReduce::argmin(this->storage_.get(), output.m_shape.shape.data(), outer, dim, inner);
    return output;
}

Tensor Tensor::matmul(const Tensor &other) const {
    const std::initializer_list output_shape = {
        this->m_shape.shape[0],
        other.m_shape.shape[1]
    };
    Tensor output(output_shape, this->device(), this->m_require || other.m_require);

    ix::TensorOp::matmul(
        this->storage_.get(),
        this->m_shape,
        other.storage_.get(),
        other.m_shape,
        output.storage_.get(),
        output.m_shape
    );

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::matmul>(this->pack(), other.pack());
    }

    return output;
}

Tensor Tensor::log() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::log(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::log>(this->pack());
    }

    return output;
}

Tensor Tensor::log2() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::log2(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::log2>(this->pack());
    }

    return output;
}


Tensor Tensor::log10() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::log10(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::log10>(this->pack());
    }

    return output;
}

Tensor Tensor::exp() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::exp(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::exp>(this->pack());
    }

    return output;
}

Tensor Tensor::exp2() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::exp2(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::exp2>(this->pack());
    }

    return output;
}

Tensor Tensor::exp10() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::exp10(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::exp10>(this->pack());
    }

    return output;
}

Tensor Tensor::pow(const f32 exp) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::pow(this->storage_.get(), exp, output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::pow>(this->pack(), exp);
    }

    return output;
}

Tensor Tensor::sqrt() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::sqrt(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sqrt>(this->pack());
    }

    return output;
}

Tensor Tensor::rsqrt() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::rsqrt(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::rsqrt>(this->pack());
    }

    return output;
}

Tensor Tensor::square() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::square(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::square>(this->pack());
    }

    return output;
}

Tensor Tensor::sin() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::sin(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sin>(this->pack());
    }

    return output;
}

Tensor Tensor::cos() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::cos(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::cos>(this->pack());
    }

    return output;
}

Tensor Tensor::tan() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::tan(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::tan>(this->pack());
    }

    return output;
}

Tensor Tensor::cot() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::cot(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::cot>(this->pack());
    }

    return output;
}

Tensor Tensor::abs() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::abs(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::abs>(this->pack());
    }

    return output;
}

Tensor Tensor::neg() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::neg(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::neg>(this->pack());
    }

    return output;
}

Tensor Tensor::sign() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::sign(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sign>(this->pack());
    }

    return output;
}

Tensor Tensor::erf() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::erf(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::erf>(this->pack());
    }

    return output;
}

Tensor Tensor::inv() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::reciprocal(this->storage_.get(), output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::inv>(this->pack());
    }

    return output;
}

Tensor Tensor::slice(i64 dim, i64 start, i64 end) const {
    const i64 ndim_actual = static_cast<i64>(this->ndim());

    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim < 0 || dim >= ndim_actual, "Dimension out of range");

    const i64 dim_len = this->m_shape.shape[dim];

    if (start < 0) {
        start += dim_len;
    }
    if (end < 0) {
        end += dim_len;
    }

    if (start < 0) {
        start = 0;
    }
    if (start > dim_len) {
        start = dim_len;
    }
    if (end < 0) {
        end = 0;
    }
    if (end > dim_len) {
        end = dim_len;
    }

    i64 sliced_len = end - start;
    if (sliced_len < 0) {
        sliced_len = 0;
    }

    auto new_shape = this->m_shape;

    new_shape.shape[dim] = sliced_len;

    new_shape.offset = this->m_shape.offset + (start * this->m_shape.stride[dim]);

    Tensor output(new_shape, this->storage_, this->m_require);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::slice>(this->pack(), dim, start, end);
    }

    return output;
}

Tensor Tensor::clamp(const f32 min, const f32 max) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::clamp(this->storage_.get(), min, max, output.storage_.get());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::clamp>(this->pack(), min, max);
    }

    return output;
}

Tensor Tensor::add(const Tensor &other) const {
    const auto _shape = broadcast_shape(this->m_shape, other.m_shape);
    Tensor output(_shape.shape, this->device(), this->m_require || other.m_require);

    ix::TensorOp::add(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::add>(this->pack(), other.pack());
    }

    return output;
}

Tensor Tensor::sub(const Tensor &other) const {
    const auto _shape = broadcast_shape(this->m_shape, other.m_shape);
    Tensor output(_shape.shape, this->device(), this->m_require || other.m_require);

    ix::TensorOp::sub(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::sub>(this->pack(), other.pack());
    }

    return output;
}

Tensor Tensor::mul(const Tensor &other) const {
    const auto _shape = broadcast_shape(this->m_shape, other.m_shape);
    Tensor output(_shape.shape, this->device(), this->m_require || other.m_require);

    ix::TensorOp::mul(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::mul>(this->pack(), other.pack());
    }

    return output;
}

Tensor Tensor::div(const Tensor &other) const {
    const auto _shape = broadcast_shape(this->m_shape, other.m_shape);
    Tensor output(_shape.shape, this->device(), this->m_require || other.m_require);

    ix::TensorOp::div(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::div>(this->pack(), other.pack());
    }

    return output;
}

Tensor Tensor::add(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::add(this->storage_.get(), value, output.storage_.get(), this->len());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::scalar_add>(this->pack());
    }

    return output;

}

Tensor Tensor::sub(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::sub(this->storage_.get(), value, output.storage_.get(), this->len());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::scalar_sub>(this->pack());
    }

    return output;
}

Tensor Tensor::mul(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::mul(this->storage_.get(), value, output.storage_.get(), this->len());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::scalar_mul>(this->pack(), value);
    }

    return output;
}

Tensor Tensor::div(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::div(this->storage_.get(), value, output.storage_.get(), this->len());

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::scalar_div>(this->pack(), value);
    }

    return output;
}

Tensor Tensor::to(const DeviceType _device) {
    this->storage_->SetDevice(_device);

    if (this->gradient_) {
        this->gradient_->storage_->SetDevice(_device);
    }

    return *this;
}

Tensor Tensor::transpose() const {
    CXM_ASSERT(this->ndim() >= 2, "Transpose requires at least 2 dimensions.");

    Tensor output(*this);

    const i32 d1 = static_cast<i32>(this->ndim()) - 1;
    const i32 d2 = static_cast<i32>(this->ndim()) - 2;

    std::swap(output.m_shape.shape[d1], output.m_shape.shape[d2]);
    std::swap(output.m_shape.stride[d1], output.m_shape.stride[d2]);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::transpose>(this->pack());
    }

    return output;
}

Tensor Tensor::permute(const std::vector<i64>& dims) const {
    const size_t ndim = this->ndim();
    CXM_ASSERT(dims.size() == ndim, "Permute dims must match tensor rank.");

    Tensor output(*this);
    std::vector seen(ndim, false);

    for (size_t i = 0; i < ndim; ++i) {
        i64 d = dims[i];
        if (d < 0) d += static_cast<i64>(ndim);

        CXM_ASSERT(d >= 0 && d < static_cast<i64>(ndim), "Dimension out of range.");
        CXM_ASSERT(!seen[d], "Duplicate dimension in permute.");
        seen[d] = true;

        output.m_shape.shape[i] = this->m_shape.shape[d];
        output.m_shape.stride[i] = this->m_shape.stride[d];
    }
    return output;
}

Tensor Tensor::reshape(const std::initializer_list<i64> _new_shape) const {
    const TensorShape _shape(_new_shape);
    Tensor output(_shape, this->storage_, this->m_require);

    if (output.m_require) {
        output.flow_ = std::make_shared<meta::reshape>(this->pack(), _new_shape);
    }

    return output;
}

Tensor Tensor::squeeze(i64 dim) const {
    Tensor output(*this);
    output.m_shape.shape.clear();
    output.m_shape.stride.clear();

    const size_t ndim = this->ndim();
    if (dim < 0) dim += static_cast<i64>(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        if (dim != -1 && static_cast<i64>(i) == dim) {
            CXM_ASSERT(this->m_shape.shape[i] != 1, "Can only squeeze dim of size 1.");
            continue;
        }
        if (dim == -1 && this->m_shape.shape[i] == 1) {
            continue;
        }

        output.m_shape.shape.push_back(this->m_shape.shape[i]);
        output.m_shape.stride.push_back(this->m_shape.stride[i]);
    }
    return output;
}

Tensor Tensor::unsqueeze(i64 dim) const {
    const size_t ndim = this->ndim();
    if (dim < 0) {
        dim += static_cast<i64>(ndim + 1);
    }

    Tensor output(*this);
    output.m_shape.shape.insert(output.m_shape.shape.begin() + dim, 1);
    output.m_shape.stride.insert(output.m_shape.stride.begin() + dim, 0);
    return output;
}

Tensor Tensor::contiguous() const {
    Tensor output(this->m_shape, this->storage_, this->m_require);

    output.flow_ = this->flow_;
    if (output.m_require) {
        output.gradient_ = this->gradient_;
    }
    return output;
}

Tensor Tensor::detach() const {
    Tensor output(*this);
    output.flow_ = nullptr;
    output.gradient_ = nullptr;
    output.m_require = false;
    return output;
}

Tensor Tensor::clone() const {
    Tensor output;
    output.m_shape = this->m_shape;
    output.storage_ = std::make_shared<TensorStorage>(this->storage_->clone());
    output.m_require = this->m_require;
    if (this->m_require) {
        output.gradient_ = std::make_shared<Tensor>(this->gradient_->clone());
    }
    return output;
}

Tensor &Tensor::grad() {
    CXM_ASSERT(this->gradient_ == nullptr, "Gradient is null");
    return *this->gradient_;
}

const Tensor &Tensor::grad() const {
    CXM_ASSERT(this->gradient_ == nullptr, "Gradient is null");
    return *this->gradient_;
}

meta::GradientPacked Tensor::pack() const {
    return {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_require};
}

void Tensor::reduce_sizes(const std::vector<i64> &dims, size_t &outer_size, size_t &dim_size, size_t &inner_size) const {
    const size_t ndim_actual = this->m_shape.shape.size();

    CXM_ASSERT(!this->is_contiguous(), "Tensor must be contiguous.");

    std::vector reduce_mask(ndim_actual, false);
    for (const i64 item : dims) {
        const i64 d = (item < 0) ? item + static_cast<i64>(ndim_actual) : item;
        CXM_ASSERT(d >= 0 && d < static_cast<i64>(ndim_actual), "Dimension out of range");
        reduce_mask[d] = true;
    }

    outer_size = 1; dim_size = 1; inner_size = 1;
    int stage = 0;

    for (size_t i = 0; i < ndim_actual; ++i) {
        const auto len = static_cast<size_t>(this->m_shape.shape[i]);
        if (reduce_mask[i]) {
            CXM_ASSERT(stage == 2, "Dimensions must be contiguous.");
            stage = 1;
            dim_size *= len;
        } else {
            if (stage == 0) outer_size *= len;
            else {
                stage = 2;
                inner_size *= len;
            }
        }
    }
}

Tensor::Tensor(TensorShape _shape, const std::shared_ptr<TensorStorage> &_storage, const bool _requires_grad) : m_shape(std::move(_shape)), m_require(_requires_grad) {
    this->storage_ = _storage;

    if (this->m_require) {
        this->gradient_ = std::make_shared<Tensor>(this->shape(), this->device());
        this->gradient_->zero();
    }
}