//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow) : GradientFlow(1) {
    this->tx_stor = tx_stor;
    this->ty_stor = ty_stor;

    this->tx_grad_stor = tx_grad_stor;
    this->ty_grad_stor = ty_grad_stor;

    this->tx_flow = tx_flow;
    this->ty_flow = ty_flow;
}

void addition::backward(MindTensor *_grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        tx.grad() += (*_grad);

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }

    const auto ty_storage = this->ty_stor.lock();
    const auto ty_grad    = this->ty_grad_stor.lock();
    const auto ty_fl      = this->ty_flow.lock();

    if (ty_storage && ty_grad) {
        MindTensor ty(ty_storage, ty_grad, ty_fl);

        ty.grad() += (*_grad);

        if (ty_fl != nullptr) {
            ty.backward(ty.grad());
        }
    }
}

subtraction::subtraction(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow) : GradientFlow(2) {
    this->tx_stor = tx_stor;
    this->ty_stor = ty_stor;

    this->tx_grad_stor = tx_grad_stor;
    this->ty_grad_stor = ty_grad_stor;

    this->tx_flow = tx_flow;
    this->ty_flow = ty_flow;
}

void subtraction::backward(MindTensor *_grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        tx.grad() += (*_grad);

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }

    const auto ty_storage = this->ty_stor.lock();
    const auto ty_grad    = this->ty_grad_stor.lock();
    const auto ty_fl      = this->ty_flow.lock();

    if (ty_storage && ty_grad) {
        MindTensor ty(ty_storage, ty_grad, ty_fl);

        ty.grad() -= (*_grad);

        if (ty_fl != nullptr) {
            ty.backward(ty.grad());
        }
    }
}