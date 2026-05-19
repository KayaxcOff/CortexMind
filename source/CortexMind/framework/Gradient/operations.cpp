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
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad;
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() += _grad;
    }

    this->propagate_backward(_grad);
}

sub::sub(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("SubBackward", 2) {
    this->tx = new Tensor(_x);
    this->ty = new Tensor(_y);
}

sub::~sub() {
    delete this->tx;
    delete this->ty;
}

void sub::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad;
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() -= _grad;
    }

    this->propagate_backward(_grad);
}

mul::mul(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("MulBackward", 3) {
    this->tx = new Tensor(_x);
    this->ty = new Tensor(_y);
}

mul::~mul() {
    delete this->tx;
    delete this->ty;
}

void mul::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * (*this->ty);
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() += _grad * (*this->tx);
    }

     this->propagate_backward(_grad);
}

div::div(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("DivBackward", 4) {
    this->tx = new Tensor(_x);
    this->ty = new Tensor(_y);
}

div::~div() {
    delete this->tx;
    delete this->ty;
}

void div::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad / (*this->ty);
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() -= _grad * (*this->tx) / ((*this->ty) * (*this->ty));
    }

    this->propagate_backward(_grad);
}

sum::sum(const GradientPacked &_x) : GradientFlow("SumBackward", 5) {
    this->tx = new Tensor(_x);
}

sum::~sum() {
    delete this->tx;
}

void sum::backward(const Tensor &_grad) {
    /*
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->tx->shape(), this->tx->device());
        ones.ones();

        this->tx->grad() += _grad * ones;
    }

    this->propagate_backward(_grad);
    */
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->tx->shape(), this->tx->device());
        ones.ones();

        const Tensor grad_expanded = _grad * ones;  // {1} * (M,N) = (M,N)

        this->tx->grad() += grad_expanded;
        this->propagate_backward(grad_expanded);  // ← Broadcast version geç!
    } else {
        this->propagate_backward(_grad);
    }

}

matmul::matmul(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("MatMulBackward", 6) {
    this->tx = new Tensor(_x);
    this->ty = new Tensor(_y);
}

matmul::~matmul() {
    delete this->tx;
    delete this->ty;
}

void matmul::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad.matmul(this->ty->transpose());
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() += this->tx->transpose().matmul(_grad);
    }

    this->propagate_backward(_grad);
}
