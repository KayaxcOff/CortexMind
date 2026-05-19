//
// Created by muham on 19.05.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

add::add(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("AddBackward", 1) {
    this->tx = new Tensor(_x);
    this->ty = new Tensor(_y);
}

add::~add() {
    delete this->tx;
    delete this->ty;
}

void add::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) {
        this->tx->grad() += _grad;
        this->tx->backward(this->tx->grad());
    }
    if (this->ty->has_grad()) {
        this->ty->grad() += _grad;
        this->ty->backward(this->ty->grad());
    }
}
