//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/matrix.hpp>
#include <CortexMind/framework/Engine/IX/scalar.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw::ix;
using namespace cortex::_fw;

Tensor Tensor::operator+(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad);

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

    return output;
}

Tensor Tensor::operator-(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad);

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

    return output;
}

Tensor Tensor::operator*(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad);

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

    return output;
}

Tensor Tensor::operator/(const Tensor &other) const {
    const auto out_shape   = broadcast_shape(this->m_shape, other.m_shape);

    Tensor output(out_shape, this->storage_->device(), this->m_requires_grad);

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

    return output;
}

Tensor &Tensor::operator+=(const Tensor &other) {

    MatrixOp::add(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides
    );

    return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {

    MatrixOp::sub(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides
    );

    return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {

    MatrixOp::div(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides
    );

    return *this;
}

Tensor &Tensor::operator/=(const Tensor &other) {

    MatrixOp::div(
        this->storage_.get(),
        this->m_shape,
        this->m_strides,
        other.storage_.get(),
        other.m_shape,
        other.m_strides
    );

    return *this;
}

Tensor Tensor::operator+(const f32 value) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ScalarOp::add(this->storage_.get(), value, output.storage_.get(), this->len());

    return output;
}

Tensor Tensor::operator-(const f32 value) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ScalarOp::sub(this->storage_.get(), value, output.storage_.get(), this->len());

    return output;
}

Tensor Tensor::operator*(const f32 value) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ScalarOp::mul(this->storage_.get(), value, output.storage_.get(), this->len());

    return output;
}

Tensor Tensor::operator/(const f32 value) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ScalarOp::div(this->storage_.get(), value, output.storage_.get(), this->len());

    return output;
}

Tensor &Tensor::operator+=(const f32 value) {
    ScalarOp::add(this->storage_.get(), value, this->len());
    return *this;
}

Tensor &Tensor::operator-=(const f32 value) {
    ScalarOp::sub(this->storage_.get(), value, this->len());
    return *this;
}

Tensor &Tensor::operator*=(const f32 value) {
    ScalarOp::mul(this->storage_.get(), value, this->len());
    return *this;
}

Tensor &Tensor::operator/=(const f32 value) {
    ScalarOp::div(this->storage_.get(), value, this->len());
    return *this;
}