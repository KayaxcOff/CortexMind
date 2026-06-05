//
// Created by muham on 4.06.2026.
//

#include "CortexMind/framework/Shape/shape.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/tensor_meta.hpp>

using namespace cortex::_fw;

TensorShape::TensorShape(std::initializer_list<i64> _shape) : shape({}), stride({}), offset(0), ndim(0) {
    CXM_ASSERT(_shape.size() > CXM_MAX_DIMS, "Tensor dimension count exceeds maximum allowed dimension (CXM_MAX_DIMS = 8).");

    this->ndim = static_cast<i32>(_shape.size());

    if (this->ndim == 0) {
        return;
    }

    std::ranges::copy(_shape, this->shape.begin());

    this->stride = compute_stride(this->shape, this->ndim);
}

TensorShape::TensorShape(const std::span<const i64> &_shape) : shape({}), stride({}), offset(0), ndim(static_cast<i32>(_shape.size())) {
    for (size_t i = 0; i < this->ndim; i++) {
        this->shape[i] = static_cast<i64>(_shape[i]);
    }
    this->stride = compute_stride(this->shape, this->ndim);
}