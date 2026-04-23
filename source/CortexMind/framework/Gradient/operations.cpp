//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::shared_ptr<TensorStorage> &tx_s, const std::shared_ptr<TensorStorage> &ty_s, MindTensor* tx_grad, MindTensor* ty_grad) : GradientFlow(1) {
    this->tx = new MindTensor(*tx_s);
    this->tx->require_grad();
    this->tx->set_grad(*tx_grad);

    this->ty = new MindTensor(*ty_s);
    this->ty->require_grad();
    this->ty->set_grad(*ty_grad);
}

addition::~addition() {
    delete this->tx;
    delete this->ty;
}

void addition::backward(MindTensor *_grad) {

    if (tx->requires_grad()) {
        tx->grad() += *_grad;
        tx->backward(tx->grad());
    }
    if (ty->requires_grad()) {
        ty->grad() += *_grad;
        ty->backward(ty->grad());
    }
}