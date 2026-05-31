//
// Created by muham on 19.05.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Engine/IX/convolution.hpp>
#include <CortexMind/framework/Tensor/tensor.hpp>
#include <algorithm>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

add::add(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("Add") {
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

sub::sub(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("Sub") {
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

mul::mul(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("Mul") {
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

div::div(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("Div") {
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

sum::sum(const GradientPacked &_x) : GradientFlow("Sum") {
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

matmul::matmul(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("MatMul") {
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

pow::pow(const GradientPacked &_x, const f32 _exp) : GradientFlow("Pow") {
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

sqrt::sqrt(const GradientPacked &_x) : GradientFlow("Sqrt") {
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

exp::exp(const GradientPacked &_x) : GradientFlow("Exp") {
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

log::log(const GradientPacked &_x) : GradientFlow("Log") {
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

rsqrt::rsqrt(const GradientPacked &_x) : GradientFlow("Rsqrt") {
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

sin::sin(const GradientPacked &_x) : GradientFlow("Sin") {
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

cos::cos(const GradientPacked &_x) : GradientFlow("Cos") {
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

abs::abs(const GradientPacked &_x) : GradientFlow("Abs") {
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
neg::neg(const GradientPacked &_x) : GradientFlow("Neg") {
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

add_scalar::add_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("AddScalar"), scalar(_scalar) {
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

sub_scalar::sub_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("SubScalar"), scalar(_scalar) {
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

mul_scalar::mul_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("MulScalar"), scalar(_scalar) {
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

div_scalar::div_scalar(const GradientPacked &_x, const f32 _scalar) : GradientFlow("DivScalar"), scalar(_scalar) {
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

relu::relu(const GradientPacked &_x) : GradientFlow("ReLU") {
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

tanh::tanh(const GradientPacked &_x, const GradientPacked& _y) : GradientFlow("Tanh") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(_y);
}

tanh::~tanh() {
    delete this->tx;
    delete this->cached_output;
}

void tanh::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->cached_output->shape(), this->cached_output->device());
        ones.ones();

        const Tensor grad_coeff = ones - this->cached_output->pow();
        const Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

sigmoid::sigmoid(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("Sigmoid") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(_y);
}

sigmoid::~sigmoid() {
    delete this->tx;
    delete this->cached_output;
}

void sigmoid::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->cached_output->shape(), this->cached_output->device());
        ones.ones();

        Tensor one_minus_output = ones - (*this->cached_output);
        Tensor grad_coeff = (*this->cached_output) * one_minus_output;
        Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

gelu::gelu(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("GeLU") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(_y);
}

gelu::~gelu() {
    delete this->tx;
    delete this->cached_output;
}

void gelu::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        constexpr f32 SQRT_2_OVER_PI = 0.7978845608f;
        constexpr f32 COEFF = 0.044715f;

        Tensor x = *this->tx;
        Tensor cdf = *this->cached_output;

        Tensor x_sq = x * x;
        Tensor pdf = (-0.5f * x_sq).exp() * 0.3989422804f;

        Tensor d_cdf_dx = pdf * SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x_sq);

        Tensor grad_coeff = cdf + x * d_cdf_dx;

        Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

leaky_relu::leaky_relu(const GradientPacked &_x, const f32 alpha) : GradientFlow("LeakyReLU") {
    this->tx = new Tensor(_x);
    this->alpha = alpha;
}

leaky_relu::~leaky_relu() {
    delete this->tx;
}

void leaky_relu::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {

        Tensor zeros(this->tx->shape(), this->tx->device());
        zeros.zero();

        Tensor mask = (*this->tx) > zeros;

        Tensor grad_coeff = mask * 1.0f + (1.0f - mask) * this->alpha;

        Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

gelu_exact::gelu_exact(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("GeLUExact") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(_y);
}

gelu_exact::~gelu_exact() {
    delete this->tx;
    delete this->cached_output;
}

void gelu_exact::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        constexpr f32 INV_SQRT_2PI = 0.39894228f;

        Tensor x_sq = this->tx->pow();
        Tensor pdf = (-0.5f * x_sq).exp() * INV_SQRT_2PI;

        Tensor grad_coeff = (*this->cached_output) + (*this->tx) * pdf;

        Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

silu::silu(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("SiLU") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(_y);
}

silu::~silu() {
    delete this->tx;
    delete this->cached_output;
}

void silu::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor ones(this->tx->shape(), this->tx->device());
        ones.ones();

        Tensor one_minus_sigmoid = ones - (*this->cached_output);

        Tensor x_times_term = (*this->tx) * one_minus_sigmoid;

        Tensor grad_coeff = ones + x_times_term;

        grad_coeff = (*this->cached_output) * grad_coeff;

        Tensor grad_expanded = _grad * grad_coeff;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

conv2d::conv2d(const GradientPacked &_x, const GradientPacked &_weight, const GradientPacked &_bias, const i64 iH, const i64 iW, const i64 kH, const i64 kW, const i64 sH, const i64 sW, const i64 pH, const i64 pW, const i64 oH, const i64 oW) : GradientFlow("Conv2D") {
    this->tx = new Tensor(_x);
    this->tw = new Tensor(_weight);
    this->tb = new Tensor(_bias);

    this->iH = iH;
    this->iW = iW;

    this->kH = kH;
    this->kW = kW;

    this->sH = sH;
    this->sW = sW;

    this->oH = oH;
    this->oW = oW;

    this->pH = pH;
    this->pW = pW;
}

conv2d::~conv2d() {
    delete this->tx;
    delete this->tb;
    delete this->tw;
}

void conv2d::backward(const Tensor &_grad) {
    const i64 N  = tx->shape()[0];
    const i64 C  = tx->shape()[1];
    const i64 oC = tw->shape()[0];

    const Tensor d_out = _grad
        .permute({1, 0, 2, 3})
        .clone()
        .reshape({oC, N * oH * oW});

    const i64 rows = C * kH * kW;
    const i64 cols = N * oH * oW;
    Tensor col({rows, cols}, tx->device(), false);
    ix::Convolution::unfold(
        tx->get(), col.get(),
        N, C, iH, iW,
        kH, kW, sH, sW, pH, pW,
        oH, oW,
        tx->device()
    );

    if (tw->has_grad()) {
        const Tensor grad_w = d_out
            .matmul(col.transpose())
            .reshape(tw->shape());
        tw->grad() += grad_w;
        //tw->backward(grad_w);
    }

    if (tb->has_grad()) {
        const Tensor grad_b = d_out.sum({1}, false);
        tb->grad() += grad_b;
        //tb->backward(grad_b);
    }

    if (tx->has_grad()) {
        const Tensor W_flat = tw->detach().reshape({oC, C * kH * kW});
        const Tensor d_col  = W_flat.transpose().matmul(d_out);

        Tensor grad_input(tx->shape(), tx->device(), false);
        ix::Convolution::fold(
            d_col.get(), grad_input.get(),
            N, C, iH, iW,
            kH, kW, sH, sW, pH, pW,
            oH, oW,
            tx->device()
        );

        tx->grad() += grad_input;
        tx->backward(grad_input);
    }
}

reshape::reshape(const GradientPacked &_x, const std::vector<i64> &shape) : GradientFlow("Reshape") {
    this->tx = new Tensor(_x);
    this->shape = shape;
}

reshape::~reshape() {
    delete this->tx;
}

void reshape::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad.clone().reshape(this->shape);

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

permute::permute(const GradientPacked &_x, const std::vector<i64> &axis) : GradientFlow("Permute") {
    this->tx = new Tensor(_x);
    this->axis = axis;
}

permute::~permute() {
    delete this->tx;
}

void permute::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor grad_expanded = _grad.permute(this->axis);

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

sum_dim::sum_dim(const GradientPacked &_x, const std::vector<i64> &dims, const bool keep) : GradientFlow("SumDim") {
    this->tx = new Tensor(_x);
    this->dims = dims;
    this->keep = keep;
}

sum_dim::~sum_dim() {
    delete this->tx;
}

void sum_dim::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        if (this->tx->has_grad()) {
            Tensor grad_expanded = _grad;

            if (!this->keep) {
                std::vector<i64> sorted_dims = this->dims;
                std::ranges::sort(sorted_dims);
                for (const i64 d : sorted_dims) {
                    grad_expanded = grad_expanded.unsqueeze(d);
                }
            }

            Tensor ones(this->tx->shape(), this->tx->device(), false);
            ones.ones();

            grad_expanded = grad_expanded * ones;
            this->tx->grad() += grad_expanded;
            this->tx->backward(grad_expanded);
        }
    }
}

clamp::clamp(const GradientPacked &_x, const f32 min_val, const f32 max_val) : GradientFlow("Clamp") {
    this->tx = new Tensor(_x);
    this->min_val = min_val;
    this->max_val = max_val;
}

clamp::~clamp() {
    delete this->tx;
}

void clamp::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        Tensor lo(this->tx->shape(), this->tx->device(), false);
        Tensor hi(this->tx->shape(), this->tx->device(), false);
        lo.fill(this->min_val);
        hi.fill(this->max_val);

        const Tensor mask = (*this->tx > lo) * (*this->tx < hi);

        const Tensor grad_expanded = _grad * mask;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

softmax::softmax(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("Softmax") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(_y);
}

softmax::~softmax() {
    delete this->tx;
    delete this->cached_output;
}

void softmax::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) [[likely]] {
        const Tensor y_times_grad = (*this->cached_output) * _grad;
        const Tensor sum_term = y_times_grad.sum({1}, true);

        const Tensor grad_expanded = (*this->cached_output) * _grad - (*this->cached_output) * sum_term;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}