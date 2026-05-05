//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &ty_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<TensorStorage> &ty_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow, const std::shared_ptr<GradientFlow> &ty_flow) : GradientFlow(1) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
    this->ty = new MindTensor(ty_stor, ty_grad_stor, ty_flow);
}

addition::~addition() {
    delete this->tx;
    delete this->ty;
}

void addition::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad);
        this->tx->backward(this->tx->grad());
    }
    if (this->ty->isGradRequired()) {
        this->ty->grad() += (*_grad);
        this->ty->backward(this->ty->grad());
    }
}

subtraction::subtraction(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &ty_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<TensorStorage> &ty_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow, const std::shared_ptr<GradientFlow> &ty_flow) : GradientFlow(2) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
    this->ty = new MindTensor(ty_stor, ty_grad_stor, ty_flow);
}

subtraction::~subtraction() {
    delete this->tx;
    delete this->ty;
}

void subtraction::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad);
        this->tx->backward(this->tx->grad());
    }
    if (this->ty->isGradRequired()) {
        this->ty->grad() -= (*_grad);
        this->ty->backward(this->ty->grad());
    }
}

multiply::multiply(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &ty_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<TensorStorage> &ty_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow, const std::shared_ptr<GradientFlow> &ty_flow) : GradientFlow(3) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
    this->ty = new MindTensor(ty_stor, ty_grad_stor, ty_flow);
}

multiply::~multiply() {
    delete this->tx;
    delete this->ty;
}

void multiply::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad) * (*this->ty);
        this->tx->backward(this->tx->grad());
    }
    if (this->ty->isGradRequired()) {
        this->ty->grad() += (*_grad) * (*this->tx);
        this->ty->backward(this->ty->grad());
    }
}

division::division(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &ty_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<TensorStorage> &ty_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow, const std::shared_ptr<GradientFlow> &ty_flow) : GradientFlow(4) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
    this->ty = new MindTensor(ty_stor, ty_grad_stor, ty_flow);
}

division::~division() {
    delete this->tx;
    delete this->ty;
}

void division::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad) / (*this->ty);
        this->tx->backward(this->tx->grad());
    }
    if (this->ty->isGradRequired()) {
        this->ty->grad() += ((*_grad) * (*this->tx) / this->ty->pow(2.0f)) * (-1.0f);
        this->ty->backward(this->ty->grad());
    }
}

scalar_additive::scalar_additive(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow) : GradientFlow(5) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
}

scalar_additive::~scalar_additive() {
    delete this->tx;
}

void scalar_additive::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad);
        this->tx->backward(this->tx->grad());
    }
}

scalar_multiply::scalar_multiply(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow, const f32 c) : GradientFlow(6), c(c) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
}

scalar_multiply::~scalar_multiply() {
    delete this->tx;
}

void scalar_multiply::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad) * this->c;
        this->tx->backward(this->tx->grad());
    }
}

dot::dot(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &ty_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<TensorStorage> &ty_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow, const std::shared_ptr<GradientFlow> &ty_flow) : GradientFlow(7) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
    this->ty = new MindTensor(ty_stor, ty_grad_stor, ty_flow);
}

dot::~dot() {
    delete this->tx;
    delete this->ty;
}

void dot::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += _grad->dot(this->ty->transpose());
        this->tx->backward(this->tx->grad());
    }
    if (this->ty->isGradRequired()) {
        this->ty->grad() += this->tx->transpose().dot((*_grad));
        this->ty->backward(this->ty->grad());
    }
}

pow::pow(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow, const f32 exp) : GradientFlow(8), exp(exp) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
}

pow::~pow() {
    delete this->tx;
}

void pow::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += this->tx->pow(this->exp - 1.0f) * this->exp * (*_grad);
        this->tx->backward(this->tx->grad());
    }
}

log::log(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow) : GradientFlow(9) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
}

log::~log() {
    delete this->tx;
}

void log::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad) / (*this->tx);
        this->tx->backward(this->tx->grad());
    }
}

exp::exp(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow) : GradientFlow(10) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
}

exp::~exp() {
    delete this->tx;
}

void exp::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        this->tx->grad() += (*_grad) * this->tx->exp();
        this->tx->backward(this->tx->grad());
    }
}

sum::sum(const std::shared_ptr<TensorStorage> &tx_stor, const std::shared_ptr<TensorStorage> &tx_grad_stor, const std::shared_ptr<GradientFlow> &tx_flow) : GradientFlow(11) {
    this->tx = new MindTensor(tx_stor, tx_grad_stor, tx_flow);
}

sum::~sum() {
    delete this->tx;
}

void sum::backward(MindTensor *_grad) {
    if (this->tx->isGradRequired()) {
        const auto value = _grad->get()[0];
        const MindTensor ones_t(this->tx->shape(), this->tx->device());
        ones_t.fill(value);

        this->tx->grad() += ones_t;
        this->tx->backward(this->tx->grad());
    }
}