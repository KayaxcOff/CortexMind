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
    if (this->tx->has_grad()) {
        const auto dims = grad_reduce_dims(this->tx->shape(), _grad.shape());
        const Tensor grad_expanded = dims.empty() ? _grad : _grad.sum(dims);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->has_grad()) {
        const auto dims = grad_reduce_dims(this->ty->shape(), _grad.shape());
        const Tensor grad_expanded = dims.empty() ? _grad : _grad.sum(dims);
        this->ty->grad() += grad_expanded;
        this->ty->backward(grad_expanded);
    }
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
    if (this->tx->has_grad()) {
        const auto dims = grad_reduce_dims(this->tx->shape(), _grad.shape());
        const Tensor grad_expanded = dims.empty() ? _grad : _grad.sum(dims);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->has_grad()) {
        const auto dims = grad_reduce_dims(this->ty->shape(), _grad.shape());
        const Tensor grad_expanded = dims.empty() ? _grad : _grad.sum(dims);
        this->ty->grad() -= grad_expanded;
        this->ty->backward(grad_expanded.neg());
    }
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
    if (this->tx->has_grad()) {
        const Tensor grad_full = _grad * (*this->ty);
        const auto dims = grad_reduce_dims(this->tx->shape(), grad_full.shape());
        const Tensor grad_expanded = dims.empty() ? grad_full : grad_full.sum(dims);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->has_grad()) {
        const Tensor grad_full = _grad * (*this->tx);
        const auto dims = grad_reduce_dims(this->ty->shape(), grad_full.shape());
        const Tensor grad_expanded = dims.empty() ? grad_full : grad_full.sum(dims);
        this->ty->grad() += grad_expanded;
        this->ty->backward(grad_expanded);
    }
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
    if (this->tx->has_grad()) {
        const Tensor grad_full = _grad / (*this->ty);
        const auto dims = grad_reduce_dims(this->tx->shape(), grad_full.shape());
        const Tensor grad_expanded = dims.empty() ? grad_full : grad_full.sum(dims);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->has_grad()) {
        const Tensor grad_full = (_grad * (*this->tx)).neg() / ((*this->ty) * (*this->ty));
        const auto dims = grad_reduce_dims(this->ty->shape(), grad_full.shape());
        const Tensor grad_expanded = dims.empty() ? grad_full : grad_full.sum(dims);
        this->ty->grad() += grad_expanded;
        this->ty->backward(grad_expanded);
    }
}

sum::sum(const GradientPacked &_x) : GradientFlow("SumBackward", 5) {
    this->tx = new Tensor(_x);
}

sum::~sum() {
    delete this->tx;
}

void sum::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->tx->shape(), this->tx->device());
        ones.ones();

        const Tensor grad_expanded = _grad * ones;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
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
        const Tensor grad_expanded = _grad.matmul(this->ty->transpose());
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->has_grad()) [[likely]] {
        const Tensor grad_expanded = this->tx->transpose().matmul(_grad);
        this->ty->grad() += grad_expanded;
        this->ty->backward(grad_expanded);
    }
}

pow::pow(const GradientPacked &_x, const f32 _exp) : GradientFlow("PowBackward", 7) {
    this->tx = new Tensor(_x);
    this->exponent = _exp;
}

pow::~pow() {
    delete this->tx;
}

void pow::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * this->tx->pow(this->exponent - 1.0f) * this->exponent;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

sqrt::sqrt(const GradientPacked &_x) : GradientFlow("SqrtBackward", 8) {
    this->tx = new Tensor(_x);
}

sqrt::~sqrt() {
    delete this->tx;
}

void sqrt::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * (0.5f / this->tx->sqrt());
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

exp::exp(const GradientPacked &_x) : GradientFlow("ExpBackward", 9) {
    this->tx = new Tensor(_x);
}

exp::~exp() {
    delete this->tx;
}

void exp::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * this->tx->exp();
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

log::log(const GradientPacked &_x) : GradientFlow("LogBackward", 10) {
    this->tx = new Tensor(_x);
}

log::~log() {
    delete this->tx;
}

void log::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad / (*this->tx);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

rsqrt::rsqrt(const GradientPacked &_x) : GradientFlow("RsqrtBackward", 11) {
    this->tx = new Tensor(_x);
}

rsqrt::~rsqrt() {
    delete this->tx;
}

void rsqrt::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * (-0.5f) / this->tx->pow(1.5f);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

sin::sin(const GradientPacked &_x) : GradientFlow("SinBackward", 12) {
    this->tx = new Tensor(_x);
}

sin::~sin() {
    delete this->tx;
}

void sin::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * this->tx->cos();
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

cos::cos(const GradientPacked &_x) : GradientFlow("CosBackward", 13) {
    this->tx = new Tensor(_x);
}

cos::~cos() {
    delete this->tx;
}

void cos::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * this->tx->sin() * (-1.0f);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

abs::abs(const GradientPacked &_x) : GradientFlow("AbsBackward", 14) {
    this->tx = new Tensor(_x);
}

abs::~abs() {
    delete this->tx;
}

void abs::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * this->tx->sign();
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}
neg::neg(const GradientPacked &_x) : GradientFlow("NegBackward", 15) {
    this->tx = new Tensor(_x);
}

neg::~neg() {
    delete this->tx;
}

void neg::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad * (-1.0f);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

add_scalar::add_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("AddScalarBackward", 17), scalar(_scalar) {
    this->tx = new Tensor(_x);
}

add_scalar::~add_scalar() {
    delete this->tx;
}

void add_scalar::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad;
        this->tx->backward(_grad);
    }
}

sub_scalar::sub_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("SubScalarBackward", 18), scalar(_scalar) {
    this->tx = new Tensor(_x);
}

sub_scalar::~sub_scalar() {
    delete this->tx;
}

void sub_scalar::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad;
        this->tx->backward(_grad);
    }
}

mul_scalar::mul_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("MulScalarBackward", 19), scalar(_scalar) {
    this->tx = new Tensor(_x);
}

mul_scalar::~mul_scalar() {
    delete this->tx;
}

void mul_scalar::backward(const Tensor &_grad) {
    const Tensor grad_expanded = _grad * this->scalar;
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

div_scalar::div_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("DivScalarBackward", 20), scalar(_scalar) {
    this->tx = new Tensor(_x);
}

div_scalar::~div_scalar() {
    delete this->tx;
}

void div_scalar::backward(const Tensor &_grad) {
    const Tensor grad_expanded = _grad / this->scalar;
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}