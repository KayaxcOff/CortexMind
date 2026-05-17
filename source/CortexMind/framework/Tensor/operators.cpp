//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/matrix.hpp>
#include <CortexMind/framework/Engine/IX/scalar.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>
#include <functional>
#include <ostream>
#include <type_traits>

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

    MatrixOp::mul(
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

Tensor &Tensor::operator=(const Tensor &other) {
    this->m_shape = other.m_shape;
    this->m_strides = other.m_strides;
    this->m_offset = other.m_offset;
    this->m_requires_grad = other.m_requires_grad;

    this->storage_ = other.storage_;
    this->flow_ = other.flow_;

    if (other.m_requires_grad) {
        this->gradient_ = other.gradient_;
    }

    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    this->m_shape = std::move(other.m_shape);
    this->m_strides = std::move(other.m_strides);
    this->m_offset = other.m_offset;
    this->m_requires_grad = other.m_requires_grad;

    this->storage_ = std::move(other.storage_);
    this->flow_ = std::move(other.flow_);

    if (other.m_requires_grad) {
        this->gradient_ = std::move(other.gradient_);
    }

    return *this;
}

namespace cortex::_fw {

    static void print_recursive(std::ostream& os, const Tensor& tensor, const size_t dim, const size_t offset) {
        const auto& shape = tensor.shape();

        if (dim == shape.size() - 1) {
            os << "[";

            for (i64 i = 0; i < shape[dim]; ++i) {
                os << tensor.get()[offset + i];

                if (i + 1 < shape[dim]) {
                    os << ", ";
                }
            }

            os << "]";
            return;
        }

        os << "[";

        size_t stride = 1;
        for (size_t i = dim + 1; i < shape.size(); ++i) {
            stride *= shape[i];
        }

        for (i64 i = 0; i < shape[dim]; ++i) {

            print_recursive(
                os,
                tensor,
                dim + 1,
                offset + i * stride
            );

            if (i + 1 < shape[dim]) {
                os << ",\n";

                for (size_t j = 0; j < dim + 1; ++j) {
                    os << " ";
                }
            }
        }

        os << "]";
    }

    Tensor operator-(const f32 value, const Tensor& tensor) {
        return tensor.neg() + value;
    }

    Tensor operator*(const f32 value, const Tensor& tensor) {
        return tensor * value;
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {

        if (tensor.empty()) {
            os << "[]";
            return os;
        }

        print_recursive(os, tensor, 0, 0);

        return os;
    }

} //namespace cortex::_fw