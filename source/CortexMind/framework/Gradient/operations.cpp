//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &ty_stor, MindTensor *tx_grad, MindTensor *ty_grad) : GradientFlow(1) {
    this->tx = new MindTensor(*tx_stor);
    this->ty = new MindTensor(*ty_stor);

    this->tx->require_grad();
    this->ty->require_grad();

    this->tx->set_grad(tx_grad);
    this->ty->set_grad(ty_grad);
}

addition::~addition() {
    delete this->tx;
    delete this->ty;
}

void addition::backward(MindTensor *_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += *_grad;
        this->tx->backward(this->tx->grad());
    }
    if (this->ty->requires_grad()) {
        this->ty->grad() += *_grad;
        this->ty->backward(this->ty->grad());
    }
}