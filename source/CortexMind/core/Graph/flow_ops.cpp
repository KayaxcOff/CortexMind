//
// Created by muham on 24.02.2026.
//

#include "CortexMind/core/Graph/flow_ops.hpp"
#include <CortexMind/core/Engine/Tensor/tensor.hpp>
#include <utility>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

[[nodiscard]]
static MindTensor reduce_to(const MindTensor& grad, const std::vector<i64>& target_shape) {
    MindTensor output = grad.clone();

    while (output.ndim() > static_cast<i64>(target_shape.size()))
        output = output.sum(0, false);

    for (i64 i = 0; i < output.ndim(); ++i) {
        if (i < static_cast<i64>(target_shape.size()) &&
            target_shape[i] == 1 && output.shape()[i] > 1) {
            output = output.sum(i, true);
            }
    }
    return output;
}

add::add(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void add::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        const MindTensor dx = (this->tx->shape() == _grad.shape()) ? _grad : reduce_to(_grad, this->tx->shape());
        this->tx->grad() += dx;
        this->tx->keep_backward(&this->tx->grad());
    }
    if (this->ty->requires_grad()) {
        const MindTensor dy = (this->ty->shape() == _grad.shape()) ? _grad : reduce_to(_grad, this->ty->shape());
        this->ty->grad() += dy;
        this->ty->keep_backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> add::inputs() {
    return {this->tx, this->ty};
}

sub::sub(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void sub::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        const MindTensor dx = (this->tx->shape() == _grad.shape()) ? _grad : reduce_to(_grad, this->tx->shape());
        this->tx->grad() += dx;
        this->tx->keep_backward(&this->tx->grad());
    }
    if (this->ty->requires_grad()) {
        const MindTensor dy = (this->ty->shape() == _grad.shape()) ? _grad : reduce_to(_grad, this->ty->shape());
        this->ty->grad() -= dy;
        this->ty->keep_backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> sub::inputs() {
    return {this->tx, this->ty};
}

mul::mul(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void mul::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        MindTensor dx = _grad * this->ty->expand(_grad.shape()).clone();
        dx = (this->tx->shape() == dx.shape()) ? dx : reduce_to(dx, this->tx->shape());
        this->tx->grad() += dx;
        this->tx->keep_backward(&this->tx->grad());
    }
    if (this->ty->requires_grad()) {
        MindTensor dy = _grad * this->tx->expand(_grad.shape()).clone();
        dy = (this->ty->shape() == dy.shape()) ? dy : reduce_to(dy, this->ty->shape());
        this->ty->grad() += dy;
        this->ty->keep_backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> mul::inputs() {
    return {this->tx, this->ty};
}

div::div(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void div::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        MindTensor b = this->ty->expand(_grad.shape()).clone();
        MindTensor dx = _grad / b;
        dx = (this->tx->shape() == dx.shape()) ? dx : reduce_to(dx, this->tx->shape());
        this->tx->grad() += dx;
        this->tx->keep_backward(&this->tx->grad());
    }
    if (this->ty->requires_grad()) {
        MindTensor a = this->tx->expand(_grad.shape()).clone();
        MindTensor b = this->ty->expand(_grad.shape()).clone();
        MindTensor dy = _grad * a / (b * b) * (-1.0f);
        dy = (this->ty->shape() == dy.shape()) ? dy : reduce_to(dy, this->ty->shape());
        this->ty->grad() += dy;
        this->ty->keep_backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> div::inputs() {
    return {this->tx, this->ty};
}

matmul::matmul(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void matmul::backward(MindTensor& _grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad.matmul(this->ty->transpose());
        this->tx->keep_backward(&this->tx->grad());
    }
    if (this->ty->requires_grad()) {
        this->ty->grad() += this->tx->transpose().matmul(_grad);
        this->ty->keep_backward(&this->ty->grad());
    }
}

std::vector<MindTensor *> matmul::inputs() {
    return {this->tx, this->ty};
}

sum::sum(MindTensor *x) : tx(x) {}

void sum::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad().fill(_grad.at(0));
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> sum::inputs() {
    return {this->tx};
}

sqrt::sqrt(MindTensor *x) : tx(x) {}

void sqrt::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad / ((this->tx->sqrt()) * 2.0f);
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> sqrt::inputs() {
    return {this->tx};
}

pow::pow(MindTensor *x, const f32 exp) : tx(x), exp(exp) {}

void pow::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad * (this->tx->pow(this->exp - 1.0f) * this->exp);
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> pow::inputs() {
    return {this->tx};
}

exp::exp(MindTensor *x) : tx(x) {}

void exp::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad * (this->tx->exp());
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> exp::inputs() {
    return {this->tx};
}

log::log(MindTensor *x) : tx(x) {}

void log::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad / (*this->tx);
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> log::inputs() {
    return {this->tx};
}

abs::abs(MindTensor *x) : tx(x) {}

void abs::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        MindTensor sign(tx->shape(), tx->devices(), false);
        const size_t num = tx->numel();
        for (size_t i = 0; i < num; ++i) {
            const f32 v = tx->get()[i];
            sign.get()[i] = (v > 0.0f) ? 1.0f : (v < 0.0f) ? -1.0f : 0.0f;
        }
        tx->grad() += _grad * sign;
        tx->keep_backward(&tx->grad());
    }
}

std::vector<MindTensor *> abs::inputs() {
    return {this->tx};
}

expand::expand(MindTensor *x, std::vector<i64> orig) : tx(x), original_shape(std::move(orig)) {}

void expand::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad.sum(0, true);
        tx->keep_backward(&tx->grad());
    }
}

std::vector<MindTensor *> expand::inputs() {
    return {this->tx};
}

repeat::repeat(MindTensor *x, const i64 t, const i64 d) : tx(x), times(t), dim(d) {}

void repeat::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad.sum(0, true);
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> repeat::inputs() {
    return {this->tx};
}

relu::relu(MindTensor *x) : tx(x) {}

void relu::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        MindTensor mask(this->tx->shape(), this->tx->devices(), false);
        const size_t num = this->tx->numel();
        for (size_t i = 0; i < num; ++i) mask.get()[i] = this->tx->get()[i] > 0.0f ? 1.0f : 0.0f;
        this->tx->grad() += _grad * mask;
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> relu::inputs() {
    return {this->tx};
}

tanh::tanh(MindTensor *x) : tx(x) {}

void tanh::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad * (this->tx->tanh().pow(2.0f) * (-1.0f) + 1.0f);
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> tanh::inputs() {
    return {this->tx};
}

sigmoid::sigmoid(MindTensor *x) : tx(x) {}

void sigmoid::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad * this->tx->sigmoid() * (this->tx->sigmoid() * (-1.0f) + 1.0f);
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> sigmoid::inputs() {
    return {this->tx};
}

dropout::dropout(MindTensor *x, MindTensor *y) : tx(x), ty(y) {}

void dropout::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        this->tx->grad() += _grad * (*this->ty);
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> dropout::inputs() {
    return {this->tx};
}

batch_norm::batch_norm(MindTensor *x, MindTensor *gamma, MindTensor *beta, MindTensor *x_hat, MindTensor *mean, MindTensor *var, const f32 eps) {
    this->tx = x;
    this->tgamma = gamma;
    this->tbeta = beta;
    this->tx_hat = x_hat;
    this->tmean = mean;
    this->tvar = var;
    this->eps = eps;
}

void batch_norm::backward(MindTensor &_grad) {
    i64 N = this->tx->shape()[0];

    if (this->tbeta->requires_grad()) {
        this->tbeta->grad() += _grad.sum(0);
        this->tbeta->keep_backward(&this->tbeta->grad());
    }

    if (this->tgamma->requires_grad()) {
        this->tgamma->grad() += (_grad * (*this->tx_hat)).sum(0);
        this->tgamma->keep_backward(&this->tgamma->grad());
    }

    if (tx->requires_grad()) {
        auto inv_std = ((*this->tvar + this->eps).sqrt());
        inv_std = inv_std * 1/10;

        auto sum_grad = _grad.sum(0);
        auto sum_grad_xhat = (_grad * (*this->tx_hat)).sum(0);

        auto dx =
            (1.0f / static_cast<f32>(N)) *
            (*this->tgamma) *
            inv_std *
            (
                static_cast<f32>(N) * _grad
                - sum_grad
                - (*this->tx_hat) * sum_grad_xhat
            );

        this->tx->grad() += dx;
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor*> batch_norm::inputs() {
    return {this->tx, this->tgamma, this->tbeta};
}

leaky_relu::leaky_relu(MindTensor *x, const f32 alpha) : tx(x), alpha(alpha) {}

void leaky_relu::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        const size_t num = this->tx->numel();
        MindTensor mask(this->tx->shape(), this->tx->devices(), false);

        for (size_t i = 0; i < num; ++i)
            mask.get()[i] = this->tx->get()[i] > 0.0f ? 1.0f : this->alpha;

        const MindTensor dx = _grad.detach() * mask;
        this->tx->grad() += dx;
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> leaky_relu::inputs() {
    return {this->tx};
}

bce::bce(MindTensor *x, MindTensor *y, const i64 n, const f32 eps) : tx(x), ty(y), n(n), eps(eps) {}

void bce::backward(MindTensor &_grad) {
    if (this->tx->requires_grad()) {
        const f32 scale = _grad.get()[0] / static_cast<f32>(this->n);
        for (i64 i = 0; i < this->n; ++i) {
            const f32 y    = this->ty->at(i, 0);
            const f32 yhat = this->tx->at(i, 0);
            this->tx->grad().at(i, 0) +=
                scale * (-(y / (yhat + this->eps)) + (1.0f - y) / (1.0f - yhat + this->eps));
        }
        this->tx->keep_backward(&this->tx->grad());
    }
}

std::vector<MindTensor *> bce::inputs() {
    return {this->tx};
}
