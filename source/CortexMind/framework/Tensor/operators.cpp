//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"

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

Tensor Tensor::operator/(const Tensor &other) const {
    return this->div(other);
}