//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>
#include <iostream>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &ty_stor, const std::shared_ptr<TensorStorage> &tx_grad, const std::shared_ptr<TensorStorage> &ty_grad, const std::shared_ptr<GradientFlow> &tx_flow, const std::shared_ptr<GradientFlow> &ty_flow) : GradientFlow(1) {
    this->tx_ = std::make_shared<MindTensor>(*tx_stor, *tx_grad, tx_flow);
    this->ty_ = std::make_shared<MindTensor>(*ty_stor, *ty_grad, ty_flow);
}

void addition::backward(MindTensor *_grad) {
    if (this->tx_->requires_grad()) {
        this->tx_->grad() += (*_grad);
        std::cout << "x" << std::endl;
        this->tx_->backward(this->tx_->grad());
    }
    if (this->ty_->requires_grad()) {
        this->ty_->grad() += (*_grad);
        std::cout << "x" << std::endl;
        this->ty_->backward(this->ty_->grad());
    }
}