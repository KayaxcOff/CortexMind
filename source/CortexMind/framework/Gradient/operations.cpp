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

relu::relu(const GradientPacked &_x) : GradientFlow("ReLUBackward", 21) {
    this->tx = new Tensor(_x);
}

relu::~relu() {
    delete this->tx;
}

void relu::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor mask(this->tx->shape(), this->tx->device(), false);
        Tensor zeros(this->tx->shape(), this->tx->device(), false);
        zeros.zero();

        mask = (*this->tx) > zeros;

        const Tensor grad_expanded = _grad * mask;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

tanh::tanh(const GradientPacked &_x, const GradientPacked& _y) : GradientFlow("TanhBackward", 22) {
    this->tx = new Tensor(_x);
    this->output = new Tensor(_y);
}

tanh::~tanh() {
    delete this->tx;
    delete this->output;
}

void tanh::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->output->shape(), this->output->device());
        ones.ones();

        const Tensor grad_coeff = ones - this->output->pow();
        const Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

sigmoid::sigmoid(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("SigmoidBackward", 23) {
    this->tx = new Tensor(_x);
    this->output = new Tensor(_y);
}

sigmoid::~sigmoid() {
    delete this->tx;
    delete this->output;
}

void sigmoid::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->output->shape(), this->output->device());
        ones.ones();

        Tensor one_minus_output = ones - (*this->output);
        Tensor grad_coeff = (*this->output) * one_minus_output;
        Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

conv2d::conv2d(const GradientPacked &_input, const GradientPacked &_kernel, const GradientPacked &_bias, const Tensor &_col, i64 iH, i64 iW, i64 kH, i64 kW, i64 sH, i64 sW, i64 pH, i64 pW) : GradientFlow("Conv2DBackward", 24) {
    this->t_kernel = new Tensor(_kernel);
    this->t_input = new Tensor(_input);
    this->t_bias = new Tensor(_bias);
    this->t_col = new Tensor(_col);

    this->iH_ = iH;
    this->iW_ = iW;

    this->kH_ = kH;
    this->kW_ = kW;

    this->sH_ = sH;
    this->sW_ = sW;

    this->pH_ = pH;
    this->pW_ = pW;
}

conv2d::~conv2d() {
    delete this->t_kernel;
    delete this->t_input;
    delete this->t_bias;
    delete this->t_col;
}

void conv2d::backward(const Tensor& _grad) {
    // _grad shape: (N, oC, oH, oW)
    const i64 N  = _grad.shape()[0];
    const i64 oC = _grad.shape()[1];
    const i64 oH = _grad.shape()[2];
    const i64 oW = _grad.shape()[3];

    const i64 C  = this->t_input->shape()[1];

    // grad_flat: (oC, N*oH*oW) — matmul için düzenle
    const Tensor grad_flat = _grad.permute({1,0,2,3})
                                  .clone()
                                  .reshape({oC, N*oH*oW});

    // ∂L/∂W = grad_flat @ col^T
    // (oC, N*oH*oW) @ (N*oH*oW, C*kH*kW) = (oC, C*kH*kW)
    if (this->t_kernel->has_grad()) {
        const Tensor dW_flat = grad_flat.matmul(
            this->t_col->transpose());
        // (oC, C*kH*kW) → (oC, C, kH, kW)
        const Tensor dW = dW_flat.reshape(this->t_kernel->shape());
        this->t_kernel->grad() += dW;
        this->t_kernel->backward(dW);
    }

    // ∂L/∂bias = sum over (N, oH, oW)
    // grad_flat: (oC, N*oH*oW) → sum over axis=1 → (oC)
    if (this->t_bias->has_grad()) {
        const Tensor db = grad_flat.sum({1}, false);
        this->t_bias->grad() += db;
        this->t_bias->backward(db);
    }

    // ∂L/∂input — col2im
    // weight_flat: (oC, C*kH*kW)
    // weight_flat^T @ grad_flat = (C*kH*kW, N*oH*oW) = col_grad
    if (this->t_input->has_grad()) {
        const Tensor weight_flat = this->t_kernel->reshape({oC, -1});
        const Tensor col_grad = weight_flat.transpose().matmul(grad_flat);

        // col2im: (C*kH*kW, N*oH*oW) → (N, C, iH, iW)
        Tensor input_grad(this->t_input->shape(),
                          this->t_input->device(), false);
        input_grad.zero();

        col2im_cpu(col_grad.get(), input_grad.get(),
                   N, C, iH_, iW_,
                   kH_, kW_,
                   sH_, sW_,
                   pH_, pW_,
                   oH, oW);

        this->t_input->grad() += input_grad;
        this->t_input->backward(input_grad);
    }
}

void conv2d::col2im_cpu(const f32 *col, f32 *input_grad, i64 N, i64 C, i64 H, i64 W, i64 kH, i64 kW, i64 sH, i64 sW, i64 pH, i64 pW, i64 oH, i64 oW) {
    for (i64 c = 0; c < C; ++c) {
        for (i64 kh = 0; kh < kH; ++kh) {
            for (i64 kw = 0; kw < kW; ++kw) {
                const i64 row = c * kH * kW + kh * kW + kw;

                for (i64 n = 0; n < N; ++n) {
                    for (i64 oh = 0; oh < oH; ++oh) {
                        for (i64 ow = 0; ow < oW; ++ow) {
                            const i64 col_idx = n * oH * oW + oh * oW + ow;

                            const i64 ih = oh * sH - pH + kh;
                            const i64 iw = ow * sW - pW + kw;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                input_grad[n*(C*H*W) + c*(H*W) + ih*W + iw]
                                    += col[row * (N*oH*oW) + col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}
