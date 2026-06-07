//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/TensorCompare/compare.hpp>
#include <CortexMind/framework/Engine/IX/TensorOp/op.hpp>
#include <CortexMind/framework/Engine/IX/scalar.hpp>

using namespace cortex::_fw;

Tensor Tensor::operator+(const Tensor &other) const {
    return this->add(other);
}

Tensor Tensor::operator-(const Tensor &other) const {
    return this->sub(other);
}

Tensor Tensor::operator*(const Tensor &other) const {
    return this->mul(other);
}

Tensor &Tensor::operator+=(const Tensor &other) {
    ix::TensorOp::add(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape);

    return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
    ix::TensorOp::sub(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape);

    return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
    ix::TensorOp::mul(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape);

    return *this;
}

Tensor &Tensor::operator/=(const Tensor &other) {
    ix::TensorOp::div(this->storage_.get(), this->m_shape, other.storage_.get(), other.m_shape);

    return *this;
}

Tensor Tensor::operator/(const Tensor &other) const {
    return this->div(other);
}

Tensor Tensor::operator+(const f32 value) const {
    return this->add(value);
}

Tensor Tensor::operator-(const f32 value) const {
    return this->sub(value);
}

Tensor Tensor::operator*(const f32 value) const {
    return this->mul(value);
}

Tensor Tensor::operator/(const f32 value) const {
    return this->div(value);
}

Tensor &Tensor::operator+=(const f32 value) {
    ix::ScalarOp::add(this->storage_.get(), value, this->len());

    return *this;
}

Tensor &Tensor::operator-=(const f32 value) {
    ix::ScalarOp::add(this->storage_.get(), value, this->len());

    return *this;
}

Tensor &Tensor::operator*=(const f32 value) {
    ix::ScalarOp::mul(this->storage_.get(), value, this->len());

    return *this;
}

Tensor &Tensor::operator/=(const f32 value) {
    ix::ScalarOp::div(this->storage_.get(), value, this->len());

    return *this;
}

Tensor Tensor::operator==(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorCompare::eq(this->storage_.get(), other.storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::operator!=(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorCompare::neq(this->storage_.get(), other.storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::operator>(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorCompare::gt(this->storage_.get(), other.storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::operator<(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorCompare::lt(this->storage_.get(), other.storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::operator>=(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorCompare::ge(this->storage_.get(), other.storage_.get(), output.storage_.get());

    return output;
}

Tensor Tensor::operator<=(const Tensor &other) const {
    Tensor output(this->shape(), this->device(), this->m_require);

    ix::TensorCompare::le(this->storage_.get(), other.storage_.get(), output.storage_.get());

    return output;
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this == &other) {
        return *this;
    }

    this->storage_ = other.storage_;
    this->m_shape = other.m_shape;
    this->m_require = other.m_require;
    this->flow_ = other.flow_;

    if (this->m_require) {
        this->gradient_ = other.gradient_;
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    this->storage_ = std::move(other.storage_);
    this->m_shape = std::move(other.m_shape);
    this->m_require = other.m_require;
    this->flow_ = std::move(other.flow_);

    if (this->m_require) {
        this->gradient_ = std::move(other.gradient_);
    }

    other.storage_ = nullptr;
    other.m_shape = {};
    other.m_require = false;
    other.flow_ = nullptr;
    other.gradient_ = nullptr;

    return *this;
}

namespace cortex::_fw {
    Tensor operator+(const f32 value, const Tensor& tensor) {
        return tensor + value;
    }
    Tensor operator-(const f32 value, const Tensor& tensor) {
        return value + tensor.neg();
    }
    Tensor operator*(const f32 value, const Tensor& tensor) {
        return tensor * value;
    }
    Tensor operator/(const f32 value, const Tensor& tensor) {
        return value * tensor.inv();
    }
} //namespace cortex::_fw