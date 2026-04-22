//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::shared_ptr<TensorStorage> &tx_s, const std::shared_ptr<TensorStorage> &ty_s) : GradientFlow(1) {
    this->tx_s = tx_s;

    this->tx_s->shape = tx_s->shape;
    this->tx_s->stride = tx_s->stride;
    this->tx_s->offset = tx_s->offset;

    this->ty_s = ty_s;

    this->ty_s->shape = ty_s->shape;
    this->ty_s->stride = ty_s->stride;
    this->ty_s->offset = ty_s->offset;
}

void addition::backward(MindTensor *_grad) {
    MindTensor tx(this->tx_s, true);
    MindTensor ty(this->ty_s, true);

    if (tx.requires_grad()) {
        tx.grad() += *_grad;
        tx.backward(tx.grad());
    }
    if (ty.requires_grad()) {
        ty.grad() += *_grad;
        ty.backward(ty.grad());
    }
}