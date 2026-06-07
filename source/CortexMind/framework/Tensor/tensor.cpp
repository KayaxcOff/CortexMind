//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/TensorInit/init.hpp>
#include <CortexMind/framework/Engine/IX/TensorOp/op.hpp>
#include <CortexMind/framework/Engine/IX/TensorReduce/reduce.hpp>
#include <CortexMind/framework/Engine/IX/TensorWise/wise.hpp>
#include <CortexMind/framework/Engine/IX/scalar.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.cuh>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>
#include <bitset>

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

    return output;
}

Tensor Tensor::mean(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;

    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::mean(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::variance() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::var(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::variance(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::var(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::stdv() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::stdv(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::stdv(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::stdv(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::norm1() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::norm1(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::norm1(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::norm1(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::norm2() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::norm2(this->storage_.get(), output.storage_.get());
    return output;
}

Tensor Tensor::norm2(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::norm2(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::max() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::max(this->storage_.get(), output.storage_.get());
    return output;
}

Tensor Tensor::max(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::max(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::min() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::min(this->storage_.get(), output.storage_.get());
    return output;
}

Tensor Tensor::min(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::min(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::sum() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::sum(this->storage_.get(), output.storage_.get());
    return output;
}

Tensor Tensor::sum(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::sum(this->storage_.get(), output.storage_.get(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::argmax() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::argmax(this->storage_.get(), output.m_shape.shape.data());
    return output;
}

Tensor Tensor::argmax(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::argmax(this->storage_.get(), output.m_shape.shape.data(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::argmin() const {
    Tensor output({1}, this->device(), this->m_require);
    ix::TensorReduce::argmin(this->storage_.get(), output.m_shape.shape.data());
    return output;
}

Tensor Tensor::argmin(const std::initializer_list<i64> dims, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    std::bitset<CXM_MAX_DIMS> reduce_mask;
    for (auto item : dims) {
        if (item < 0) item += ndim_actual;
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dims, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (reduce_mask.test(i)) {
            if (keep_dim) {
                new_shape_array[new_ndim++] = 1;
            }
        } else {
            new_shape_array[new_ndim++] = this->m_shape.shape[i];
        }
    }

    if (new_ndim == 0) {
        new_shape_array[0] = 1;
        new_ndim = 1;
    }

    const std::span<const i64> shape_span(new_shape_array.data(), new_ndim);
    Tensor output(shape_span, this->device(), this->m_require);

    ix::TensorReduce::argmin(this->storage_.get(), output.m_shape.shape.data(), outer_size, dim_size, inner_size);
    return output;
}

Tensor Tensor::matmul(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorOp::matmul(
        this->storage_.get(),
        this->m_shape,
        other.storage_.get(),
        other.m_shape,
        output.storage_.get(),
        output.m_shape
    );

    return output;
}

Tensor Tensor::log() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::log(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::log2() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::log2(this->storage_.get(), output.storage_.get());

    return output;
}


Tensor Tensor::log10() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::log10(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::exp() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::exp(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::exp2() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::exp2(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::exp10() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::exp10(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::pow(const f32 exp) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::pow(this->storage_.get(), exp, output.storage_.get());

    return output;
}

Tensor Tensor::sqrt() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::sqrt(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::rsqrt() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::rsqrt(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::square() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::square(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::sin() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::sin(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::cos() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::cos(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::tan() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::tan(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::cot() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::cot(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::abs() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::abs(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::neg() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::neg(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::sign() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::sign(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::erf() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::erf(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::inv() const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::reciprocal(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::slice(i64 dim, i64 start, i64 end) const {
    const i64 ndim_actual = this->m_shape.ndim;

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

    Tensor output(new_shape.shape, this->storage_, this->m_require);

    return output;
}

Tensor Tensor::clamp(const f32 min, const f32 max) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorWise::clamp(this->storage_.get(), min, max, output.storage_.get());

    return output;
}

Tensor Tensor::add(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorOp::add(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    return output;
}

Tensor Tensor::sub(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorOp::sub(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    return output;
}

Tensor Tensor::mul(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorOp::mul(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    return output;
}

Tensor Tensor::div(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorOp::div(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape, output.storage_.get(), output.m_shape);

    return output;
}

Tensor Tensor::add(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::add(this->storage_.get(), value, output.storage_.get(), this->len());

    return output;

}

Tensor Tensor::sub(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::sub(this->storage_.get(), value, output.storage_.get(), this->len());

    return output;
}

Tensor Tensor::mul(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::mul(this->storage_.get(), value, output.storage_.get(), this->len());

    return output;
}

Tensor Tensor::div(const f32 value) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::ScalarOp::div(this->storage_.get(), value, output.storage_.get(), this->len());

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
    CXM_ASSERT(this->m_shape.ndim >= 2, "Transpose requires at least 2 dimensions.");

    Tensor output(*this);

    const i32 d1 = this->m_shape.ndim - 1;
    const i32 d2 = this->m_shape.ndim - 2;

    std::swap(output.m_shape.shape[d1], output.m_shape.shape[d2]);
    std::swap(output.m_shape.stride[d1], output.m_shape.stride[d2]);

    return output;
}

Tensor Tensor::permute(const std::initializer_list<i64> dims) const {
    CXM_ASSERT(dims.size() == static_cast<size_t>(this->m_shape.ndim), "Permute dimensions must match tensor ndim.");

    Tensor output(*this);

    bool seen[CXM_MAX_DIMS] = { false };
    size_t idx = 0;

    for (i64 item : dims) {
        if (item < 0) {
            item += this->m_shape.ndim;
        }

        CXM_ASSERT(item >= 0 && item < this->m_shape.ndim, "Permute dimension out of range.");
        CXM_ASSERT(!seen[item], "Permute dimensions cannot contain duplicates.");
        seen[item] = true;

        output.m_shape.shape[idx] = this->m_shape.shape[item];
        output.m_shape.stride[idx] = this->m_shape.stride[item];
        idx++;
    }

    return output;
}

Tensor Tensor::reshape(const std::initializer_list<i64> _new_shape) const {
    return {_new_shape, this->storage_, this->m_require};
}

Tensor Tensor::squeeze(i64 dim) const {
    Tensor output(*this);
    output.m_shape.ndim = 0;

    if (dim < 0 && dim != -1) dim += this->m_shape.ndim;

    for (i32 i = 0; i < this->m_shape.ndim; ++i) {
        if (dim == -1) {
            if (m_shape.shape[i] == 1) {
                continue;
            }
        } else {
            if (i == static_cast<i32>(dim)) {
                CXM_ASSERT(this->m_shape.shape[i] == 1, "Can only squeeze a dimension of size 1.");
                continue;
            }
        }

        output.m_shape.shape[output.m_shape.ndim] = this->m_shape.shape[i];
        output.m_shape.stride[output.m_shape.ndim] = this->m_shape.stride[i];
        output.m_shape.ndim++;
    }

    return output;
}

Tensor Tensor::unsqueeze(i64 dim) const {
    if (dim < 0) dim += this->m_shape.ndim + 1;
    CXM_ASSERT(dim >= 0 && dim <= this->m_shape.ndim, "Unsqueeze dimension out of range.");
    CXM_ASSERT(this->m_shape.ndim < CXM_MAX_DIMS, "Max dimensions exceeded.");

    Tensor output(*this);
    output.m_shape.ndim = this->m_shape.ndim + 1;

    i32 src_idx = 0;
    for (i32 i = 0; i < output.m_shape.ndim; ++i) {
        if (i == static_cast<i32>(dim)) {
            output.m_shape.shape[i] = 1;
            output.m_shape.stride[i] = (i < output.m_shape.ndim - 1) ? this->m_shape.stride[src_idx] : 1;
        } else {
            output.m_shape.shape[i] = this->m_shape.shape[src_idx];
            output.m_shape.stride[i] = this->m_shape.stride[src_idx];
            src_idx++;
        }
    }

    return output;
}

Tensor Tensor::contiguous() const {
    return {this->shape(), this->storage_, this->m_require};
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

void Tensor::reduce_sizes(const std::initializer_list<i64> dims, size_t &outer_size, size_t &dim_size, size_t &inner_size) const {
    const i64 ndim_actual = this->m_shape.ndim;

    CXM_ASSERT(!this->is_contiguous(), "Tensor must be contiguous for multi-axis reduction.");

    std::bitset<CXM_MAX_DIMS> reduce_mask;

    for (auto item : dims) {
        if (item < 0) {
            item += ndim_actual;
        }
        CXM_ASSERT(item < 0 || item >= ndim_actual, "Dimension out of range");
        reduce_mask.set(item);
    }

    if (dims.size() == 0) {
        outer_size = 1;
        dim_size = this->len();
        inner_size = 1;
        return;
    }

    outer_size = 1;
    dim_size = 1;
    inner_size = 1;

    i32 stage = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        const auto current_dim_len = static_cast<size_t>(this->m_shape.shape[i]);

        if (reduce_mask.test(i)) {
            if (stage == 0) {
                stage = 1;
            } else if (stage == 2) {
                CXM_ASSERT(true, "Selected reduce dimensions must be contiguous (e.g., {1, 2} is valid, {0, 2} is invalid).");
            }
            dim_size *= current_dim_len;
        } else {
            if (stage == 0) {
                outer_size *= current_dim_len;
            } else {
                stage = 2;
                inner_size *= current_dim_len;
            }
        }
    }
}

Tensor::Tensor(const std::span<const i64> &_shape, const std::shared_ptr<TensorStorage> &_storage, const bool _requires_grad) : m_shape(_shape), m_require(_requires_grad) {
    this->storage_ = _storage;
}
