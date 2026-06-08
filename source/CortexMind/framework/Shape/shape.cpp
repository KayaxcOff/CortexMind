//
// Created by muham on 4.06.2026.
//

#include "CortexMind/framework/Shape/shape.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw;

TensorShape::TensorShape() : shape({}), stride({}), offset(0) {}

TensorShape::TensorShape(const std::initializer_list<i64> shape) : shape(shape), offset(0) {
    this->stride = compute_stride(this->shape);
}

TensorShape::TensorShape(const std::vector<i64> &shape) : shape(shape), offset(0) {
    this->stride = compute_stride(this->shape);
}

TensorShape::TensorShape(const std::vector<i64> &shape, const std::vector<i64> &strides) : shape(shape), stride(strides), offset(0) {}