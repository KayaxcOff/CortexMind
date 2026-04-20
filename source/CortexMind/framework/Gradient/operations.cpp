//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::shared_ptr<TensorStorage> &tx_s, const std::shared_ptr<TensorStorage> &ty_s) : GradientFlow(1) {
    this->tx = std::make_shared<MindTensor>(tx_s);
    this->ty = std::make_shared<MindTensor>(ty_s);
}

void addition::backward(MindTensor *_grad) {
    auto tx_lock = this->tx.lock();
    if (tx_lock->requires_grad()) {
        tx_lock->grad() += *_grad;
        tx_lock->backward(*_grad);
    }
    auto ty_lock = this->ty.lock();
    if (ty_lock->requires_grad()) {
        ty_lock->grad() += *_grad;
        ty_lock->backward(*_grad);
    }
}
