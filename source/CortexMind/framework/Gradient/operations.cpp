//
// Created by muham on 19.05.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>
#include <iostream>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

add::add(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("AddBackward", 1) {
    this->tx = new Tensor(_x);
    this->ty = new Tensor(_y);
    std::cout << this->name() << " initialized" << std::endl;
}

add::~add() {
    delete this->tx;
    delete this->ty;
    std::cout << this->name() << " destroyed" << std::endl;
}

void add::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad;
        //this->tx->backward(_grad);
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() += _grad;
        //this->ty->backward(_grad);
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
    std::cout << this->name() << " initialized" << std::endl;
}

mul::~mul() {
    delete this->tx;
    delete this->ty;
    std::cout << this->name() << " destroyed" << std::endl;
}

void mul::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * (*this->ty);
        //this->tx->backward(_grad);
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() += _grad * (*this->tx);
        //this->ty->backward(_grad);
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
    std::cout << this->name() << " initialized" << std::endl;
}

sum::~sum() {
    delete this->tx;
    std::cout << this->name() << " destroyed" << std::endl;
}

void sum::backward(const Tensor &_grad) {

    // if (this->tx->has_grad()) [[likely]] {
    //     Tensor ones(this->tx->shape(), this->tx->device());
    //     ones.ones();
    //
    //     const auto grad_expanded = _grad * ones;
    //
    //     this->tx->grad() += grad_expanded;
    //     this->tx->backward(this->tx->grad());
    // }

    //this->propagate_backward(_grad);

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
        //this->tx->backward(this->tx->grad());
    }
    if (this->ty->has_grad()) [[likely]] {
        this->ty->grad() += this->tx->transpose().matmul(_grad);
        //this->ty->backward(this->ty->grad());
    }
    this->propagate_backward(_grad);
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
        this->tx->grad() += _grad * this->tx->pow(this->exponent - 1.0f) * this->exponent;
    }

    this->propagate_backward(_grad);
}

sqrt::sqrt(const GradientPacked &_x) : GradientFlow("SqrtBackward", 8) {
    this->tx = new Tensor(_x);
}

sqrt::~sqrt() {
    delete this->tx;
}

void sqrt::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad / this->tx->sqrt() * 2.0f;
    }

    this->propagate_backward(_grad);
}

exp::exp(const GradientPacked &_x) : GradientFlow("ExpBackward", 9) {
    this->tx = new Tensor(_x);
}

exp::~exp() {
    delete this->tx;
}

void exp::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * this->tx->exp();
    }

    this->propagate_backward(_grad);
}

log::log(const GradientPacked &_x) : GradientFlow("LogBackward", 10) {
    this->tx = new Tensor(_x);
}

log::~log() {
    delete this->tx;
}

void log::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad / (*this->tx);
    }

    this->propagate_backward(_grad);
}

rsqrt::rsqrt(const GradientPacked &_x) : GradientFlow("RsqrtBackward", 11) {
    this->tx = new Tensor(_x);
}

rsqrt::~rsqrt() {
    delete this->tx;
}

void rsqrt::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * (-1.0f) / this->tx->pow(1.5f) * 2.0f;
    }

    this->propagate_backward(_grad);
}

sin::sin(const GradientPacked &_x) : GradientFlow("SinBackward", 12) {
    this->tx = new Tensor(_x);
}

sin::~sin() {
    delete this->tx;
}

void sin::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * this->tx->cos();
    }

    this->propagate_backward(_grad);
}

cos::cos(const GradientPacked &_x) : GradientFlow("CosBackward", 13) {
    this->tx = new Tensor(_x);
}

cos::~cos() {
    delete this->tx;
}

void cos::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * this->tx->sin() * (-1.0f);
    }

    this->propagate_backward(_grad);
}

abs::abs(const GradientPacked &_x) : GradientFlow("AbsBackward", 14) {
    this->tx = new Tensor(_x);
}

abs::~abs() {
    delete this->tx;
}

void abs::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * this->tx->sign();
    }

    this->propagate_backward(_grad);
}
neg::neg(const GradientPacked &_x) : GradientFlow("NegBackward", 15) {
    this->tx = new Tensor(_x);
}

neg::~neg() {
    delete this->tx;
}

void neg::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += _grad * (-1.0f);
    }

    this->propagate_backward(_grad);
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
    }
    this->propagate_backward(_grad);
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
    }
    this->propagate_backward(_grad);
}

mul_scalar::mul_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("MulScalarBackward", 19), scalar(_scalar) {
    this->tx = new Tensor(_x);
}

mul_scalar::~mul_scalar() {
    delete this->tx;
}

void mul_scalar::backward(const Tensor &_grad) {
    const Tensor grad_input = _grad * this->scalar;
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += grad_input;
    }
    this->propagate_backward(grad_input);
}

div_scalar::div_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("DivScalarBackward", 20), scalar(_scalar) {
    this->tx = new Tensor(_x);
}

div_scalar::~div_scalar() {
    delete this->tx;
}

void div_scalar::backward(const Tensor &_grad) {
    const Tensor grad_input = _grad / this->scalar;
    if (this->tx->has_grad()) [[likely]] {
        this->tx->grad() += grad_input;
    }
    this->propagate_backward(grad_input);
}