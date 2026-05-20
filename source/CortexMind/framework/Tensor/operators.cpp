//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#include <CortexMind/framework/Engine/IX/matrix.hpp>
#include <CortexMind/framework/Engine/IX/scalar.hpp>
#include <CortexMind/framework/Gradient/operations.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <functional>
#include <ostream>
#include <type_traits>

using namespace cortex::_fw::ix;
using namespace cortex::_fw;

namespace {
    /**
     * @brief Recursively prints a multidimensional tensor in a nested array format.
     *
     * This function traverses the tensor dimensions recursively and writes the
     * tensor contents into the given output stream using bracket notation.
     * Each dimension is represented as a nested list, similar to Python-style arrays.
     *
     * The recursion terminates when the last dimension is reached, where the
     * actual tensor elements are printed sequentially.
     *
     * @param os Output stream used to print the tensor contents.
     * @param tensor Tensor instance containing the data and shape information.
     * @param dim Current dimension being processed.
     * @param offset Linear memory offset corresponding to the current sub-tensor.
     *
     * @note The function assumes that the tensor data is stored contiguously
     *       in row-major order.
     *
     * @warning The caller must ensure that `dim` and `offset` are valid
     *          with respect to the tensor shape and underlying storage.
     */
    void print_recursive(std::ostream& os, const Tensor& tensor, const size_t dim, const size_t offset) {
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
} // unnamed namespace

Tensor Tensor::operator+(const Tensor &other) const {
    return this->addition(other);
}

Tensor Tensor::operator-(const Tensor &other) const {
    return this->subtract(other);
}

Tensor Tensor::operator*(const Tensor &other) const {
    return this->multiply(other);
}

Tensor Tensor::operator/(const Tensor &other) const {
    return this->divide(other);
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

    if (output.m_requires_grad) {

        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::add_scalar>(x, value);

        output.flow_->save(this->flow_);
    }

    return output;
}

Tensor Tensor::operator-(const f32 value) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ScalarOp::sub(this->storage_.get(), value, output.storage_.get(), this->len());

    if (output.m_requires_grad) {

        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::sub_scalar>(x, value);

        output.flow_->save(this->flow_);
    }

    return output;
}

Tensor Tensor::operator*(const f32 value) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ScalarOp::mul(this->storage_.get(), value, output.storage_.get(), this->len());

    if (output.m_requires_grad) {

        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::mul_scalar>(x, value);

        output.flow_->save(this->flow_);
    }

    return output;
}

Tensor Tensor::operator/(const f32 value) const {
    Tensor output(this->m_shape, this->storage_->device(), this->m_requires_grad);

    ScalarOp::div(this->storage_.get(), value, output.storage_.get(), this->len());

    if (output.m_requires_grad) {

        meta::GradientPacked x {this->storage_, this->flow_, this->gradient_, this->m_shape, this->m_requires_grad};

        output.flow_ = std::make_shared<meta::div_scalar>(x, value);

        output.flow_->save(this->flow_);
    }

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

bool Tensor::operator==(const Tensor &other) const {
    if (this->storage_.get() == other.storage_.get()) {
        return true;
    }
    return false;
}

bool Tensor::operator!=(const Tensor &other) const {
    return !(*this == other);
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this == &other) {
        return *this;
    }

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