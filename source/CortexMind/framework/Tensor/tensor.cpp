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

Tensor Tensor::mean() const {
    Tensor output({1}, this->device(), this->m_require);

    ix::TensorReduce::mean(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::mean(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::variance(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::stdv(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::norm1(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::norm2(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::max(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::min(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::sum(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::argmax(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

Tensor Tensor::argmin(i64 dim, const bool keep_dim) const {
    const i64 ndim_actual = this->m_shape.ndim;
    if (dim < 0) {
        dim += ndim_actual;
    }
    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    size_t outer_size, dim_size, inner_size;
    this->reduce_sizes(dim, outer_size, dim_size, inner_size);

    std::array<i64, CXM_MAX_DIMS> new_shape_array{};
    size_t new_ndim = 0;

    for (i64 i = 0; i < ndim_actual; ++i) {
        if (i == dim) {
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

void Tensor::reduce_sizes(i64 dim, size_t &outer_size, size_t &dim_size, size_t &inner_size) const {
    const i64 ndim_actual = this->m_shape.ndim;

    if (dim < 0) {
        dim += ndim_actual;
    }

    CXM_ASSERT(dim >= 0 && dim < ndim_actual, "Dimension out of range");

    outer_size = 1;
    dim_size = static_cast<size_t>(this->m_shape.shape[dim]);
    inner_size = 1;

    for (i64 i = 0; i < dim; ++i) {
        outer_size *= this->m_shape.shape[i];
    }

    for (i64 i = dim + 1; i < ndim_actual; ++i) {
        inner_size *= this->m_shape.shape[i];
    }
}


Tensor Tensor::matmul(const Tensor &other) const {

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

    ix::TensorWise::inverse(this->storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::slice(i64 dim, i64 start, i64 end) const {

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
