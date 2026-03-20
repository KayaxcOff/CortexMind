//
// Created by muham on 16.03.2026.
//

#include "CortexMind/core/Graph/ops.hpp"
#include <CortexMind/core/Engine/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

add::add(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void add::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad;
        this->tx->backward(&this->tx->grad());
    }
    if (this->ty->grad_required()) {
        this->ty->grad() += _grad;
        this->ty->backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> add::inputs() {
    return {this->tx, this->ty};
}

sub::sub(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void sub::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad;
        this->tx->backward(&this->tx->grad());
    }
    if (this->ty->grad_required()) {
        this->ty->grad() -= _grad;
        this->ty->backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> sub::inputs() {
    return {this->tx, this->ty};
}

mul::mul(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void mul::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad * (*this->ty);
        this->tx->backward(&this->tx->grad());
    }
    if (this->ty->grad_required()) {
        this->ty->grad() += _grad * (*this->tx);
        this->ty->backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> mul::inputs() {
    return {this->tx, this->ty};
}

div::div(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void div::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad / (*this->ty);
        this->tx->backward(&this->tx->grad());
    }
    if (this->ty->grad_required()) {
        this->ty->grad() -= _grad * (*this->tx) / ((*this->ty) * (*this->ty));
        this->ty->backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> div::inputs() {
    return {this->tx, this->ty};
}

matmul::matmul(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void matmul::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad.matmul(this->ty->transpose());
        this->tx->backward(&this->tx->grad());
    }
    if (this->ty->grad_required()) {
        this->ty->grad() += this->tx->transpose().matmul(_grad);
        this->ty->backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> matmul::inputs() {
    return {this->tx, this->ty};
}

sqrt::sqrt(MindTensor *x) : tx(x) {}

void sqrt::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad / ((this->tx->sqrt()) * 2.0f);
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> sqrt::inputs() {
    return {this->tx};
}

pow::pow(MindTensor *x, const f32 alpha) : tx(x), alpha(alpha) {}

void pow::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad * (this->tx->pow(this->alpha - 1.0f) * this->alpha);
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> pow::inputs() {
    return {this->tx};
}

exp::exp(MindTensor *x) : tx(x) {}

void exp::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        // d/dx e^x = e^x
        this->tx->grad() += _grad * this->tx->exp();
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> exp::inputs() {
    return {this->tx};
}

log::log(MindTensor *x) : tx(x) {}

void log::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        // d/dx ln(x) = 1/x
        this->tx->grad() += _grad / (*this->tx);
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> log::inputs() {
    return {this->tx};
}

abs::abs(MindTensor *x) : tx(x) {}

void abs::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad * (*this->tx) / this->tx->abs();
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> abs::inputs() {
    return {this->tx};
}

sum_all::sum_all(MindTensor *x) : tx(x) {}

void sum_all::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad().fill(_grad.get()[0]);
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> sum_all::inputs() { return {this->tx}; }

sum_dim::sum_dim(MindTensor *x, const i64 dim, const bool keep) : tx(x), dim(dim), keep(keep) {}

void sum_dim::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        MindTensor g = this->keep ? _grad : _grad.unsqueeze(this->dim);
        this->tx->grad() += g + MindTensor(this->tx->shape(), this->tx->device());
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> sum_dim::inputs() { return {this->tx}; }

add_scalar::add_scalar(MindTensor *x, const f32 scalar) : tx(x), scalar(scalar) {}

void add_scalar::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad;
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> add_scalar::inputs() { return {this->tx}; }

sub_scalar::sub_scalar(MindTensor *x, const f32 scalar) : tx(x), scalar(scalar) {}

void sub_scalar::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad;
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> sub_scalar::inputs() { return {this->tx}; }

mul_scalar::mul_scalar(MindTensor *x, const f32 scalar) : tx(x), scalar(scalar) {}

void mul_scalar::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad * this->scalar;
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> mul_scalar::inputs() { return {this->tx}; }

div_scalar::div_scalar(MindTensor *x, const f32 scalar) : tx(x), scalar(scalar) {}

void div_scalar::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        this->tx->grad() += _grad / this->scalar;
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> div_scalar::inputs() { return {this->tx}; }

relu::relu(MindTensor *x) : tx(x) {}

void relu::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        MindTensor mask(this->tx->shape(), this->tx->device(), false);
        const size_t num = this->tx->numel();
        for (size_t i = 0; i < num; ++i)
            mask.get()[i] = this->tx->get()[i] > 0.0f ? 1.0f : 0.0f;
        this->tx->grad() += _grad * mask;
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> relu::inputs() {
    return {this->tx};
}

leaky_relu::leaky_relu(MindTensor *x, const f32 alpha) : tx(x), alpha(alpha) {}

void leaky_relu::backward(MindTensor &_grad) {
    if (this->tx->grad_required()) {
        const size_t num = this->tx->numel();

        MindTensor mask(this->tx->shape(), this->tx->device(), false);

        for (size_t i = 0; i < num; ++i)
            mask.get()[i] = this->tx->get()[i] > 0.0f ? 1.0f : this->alpha;

        const MindTensor dx = _grad.detach() * mask;
        this->tx->grad() += dx;
        this->tx->backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> leaky_relu::inputs() {
    return {this->tx};
}
