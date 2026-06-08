//
// Created by muham on 19.05.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Engine/IX/convolution.hpp>
#include <CortexMind/framework/Tensor/tensor.hpp>
#include <cmath>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

namespace {
    std::vector<i64> grad_reduce_dims(const std::vector<i64>& input_shape, const std::vector<i64>& grad_shape) {
    std::vector<i64> output;

    if (input_shape.size() == grad_shape.size()) {
        for (size_t d = 0; d < grad_shape.size(); ++d) {
            if (input_shape[d] == 1 && grad_shape[d] > 1) {
                output.push_back(static_cast<i64>(d));
            }
        }
    } else if (grad_shape.size() > input_shape.size()) {
        const i64 offset = static_cast<i64>(grad_shape.size() - input_shape.size());

        for (i64 d = 0; d < offset; ++d) {
            output.push_back(d);
        }

        for (i64 d = offset; d < static_cast<i64>(grad_shape.size()); ++d) {
            if (const i64 input_idx = d - offset; input_shape[input_idx] == 1 && grad_shape[d] > 1) {
                output.push_back(d);
            }
        }
    }

    return output;
}

    Tensor broadcast_and_reduce_grad(const Tensor& grad, const std::vector<i64>& target_shape, sys::DeviceType device){
        //auto grad_shape = grad.shape();
        const std::vector<i64> reduce_dims = grad_reduce_dims(target_shape, grad.shape());

        if (reduce_dims.empty()) {
        }

        Tensor output = grad;
        for (auto it = reduce_dims.rbegin(); it != reduce_dims.rend(); ++it) {
            output = output.sum({*it}, false);
        }

        if (reduce_dims.size() == target_shape.size()) {
            //result = result.reshape({target_shape.begin(), target_shape.end()});
        }

        return output;
    }
} //unnamed namespace

add::add(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("Add") {
    this->tx = new Tensor(_x);
    this->ty = new Tensor(_y);
}

add::~add() {
    delete this->tx;
    delete this->ty;
}

void add::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor grad_expanded = broadcast_and_reduce_grad(_grad, this->tx->shape(), this->tx->device());
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->is_require()) {
        const Tensor grad_expanded = broadcast_and_reduce_grad(_grad, this->ty->shape(), this->ty->device());
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
    if (this->tx->is_require()) {
        const Tensor grad_expanded = broadcast_and_reduce_grad(_grad, this->tx->shape(), this->tx->device());
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->is_require()) {
        const Tensor grad_expanded = broadcast_and_reduce_grad(_grad, this->ty->shape(), this->ty->device());
        this->ty->grad() -= grad_expanded;
        this->ty->backward(grad_expanded);
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
    if (this->tx->is_require()) {
        Tensor grad_expanded = _grad * (*this->ty);
        grad_expanded = broadcast_and_reduce_grad(grad_expanded, this->tx->shape(), this->tx->device());
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->is_require()) {
        Tensor grad_expanded = _grad * (*this->tx);
        grad_expanded = broadcast_and_reduce_grad(grad_expanded, this->ty->shape(), this->ty->device());
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
    if (this->tx->is_require()) {
        Tensor grad_expanded = _grad / (*this->ty);
        grad_expanded = broadcast_and_reduce_grad(grad_expanded, this->tx->shape(), this->tx->device());
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
    if (this->ty->has_grad()) {
        Tensor grad_expanded = (_grad * (*this->tx)).neg() / this->ty->square();
        grad_expanded = broadcast_and_reduce_grad(grad_expanded, this->ty->shape(), this->ty->device());
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
    if (this->tx->is_require()) {
        const Tensor ones(this->tx->shape(), this->tx->device());
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

pow::pow(const GradientPacked &_x, const f32 exp) : GradientFlow("Pow") {
    this->tx = new Tensor(_x);
    this->exp = exp;
}

pow::~pow() {
    delete this->tx;
}

void pow::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad * (this->tx->pow(this->exp - 1.0f) * this->exp);
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
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad * (0.5f * this->tx->rsqrt());
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
    if (this->tx->is_require()) [[likely]] {
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
    if (this->tx->is_require()) [[likely]] {
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
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad * (-0.5f / this->tx->pow(1.5f));
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
    if (this->tx->is_require()) [[likely]] {
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
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad * (this->tx->sin() * (-1.0f));
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
    if (this->tx->is_require()) [[likely]] {
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
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad * (-1.0f);
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

scalar_add::scalar_add(const GradientPacked &_x) : GradientFlow("Scalar Add") {
    this->tx = new Tensor(_x);
}

scalar_add::~scalar_add() {
    delete this->tx;
}

void scalar_add::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        this->tx->grad() += _grad;
        this->tx->backward(_grad);
    }
}

scalar_sub::scalar_sub(const GradientPacked &_x) : GradientFlow("Scalar Sub") {
    this->tx = new Tensor(_x);
}

scalar_sub::~scalar_sub() {
    delete this->tx;
}

void scalar_sub::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        this->tx->grad() += _grad;
        this->tx->backward(_grad);
    }
}

scalar_mul::scalar_mul(const GradientPacked &_x, const f32 value) : GradientFlow("Scalar Mul") {
    this->tx = new Tensor(_x);
    this->value = value;
}

scalar_mul::~scalar_mul() {
    delete this->tx;
}

void scalar_mul::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad * this->value;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

scalar_div::scalar_div(const GradientPacked &_x, const f32 value) : GradientFlow("Scalar Div") {
    this->tx = new Tensor(_x);
    this->value = value;
}

scalar_div::~scalar_div() {
    delete this->tx;
}

void scalar_div::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad / this->value;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

reshape::reshape(const GradientPacked &_x, const std::initializer_list<i64> shape) : GradientFlow("Reshape") {
    this->tx = new Tensor(_x);
    this->shape = shape;
}

reshape::~reshape() {
    delete this->tx;
}

void reshape::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad.contiguous().reshape(this->shape);

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

permute::permute(const GradientPacked &_x, const std::initializer_list<i64> dims) : GradientFlow("Permute") {
    this->tx = new Tensor(_x);
    this->dims = dims;
}

permute::~permute() {
    delete this->tx;
}

void permute::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const size_t ndim = this->dims.size();

        std::vector<i64> inv_dims;

        size_t idx = 0;
        for (i64 d : this->dims) {
            if (d < 0) {
                d += static_cast<i64>(ndim);
            }
            inv_dims[d] = static_cast<i64>(idx++);
        }

        Tensor grad_expanded;
        switch (ndim) {
            case 1: grad_expanded = _grad.permute({inv_dims[0]}); break;
            case 2: grad_expanded = _grad.permute({inv_dims[0], inv_dims[1]}); break;
            case 3: grad_expanded = _grad.permute({inv_dims[0], inv_dims[1], inv_dims[2]}); break;
            case 4: grad_expanded = _grad.permute({inv_dims[0], inv_dims[1], inv_dims[2], inv_dims[3]}); break;
            case 5: grad_expanded = _grad.permute({inv_dims[0], inv_dims[1], inv_dims[2], inv_dims[3], inv_dims[4]}); break;

            default:
                CXM_ASSERT(true, "Unsupported dimension size in permute backward");
        }

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

clamp::clamp(const GradientPacked &_x, const f32 min, const f32 max) : GradientFlow("Clamp") {
    this->tx = new Tensor(_x);
    this->min = min;
    this->max = max;
}

clamp::~clamp() {
    delete this->tx;
}

void clamp::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor lo(this->tx->shape(), this->tx->device(), false);
        const Tensor hi(this->tx->shape(), this->tx->device(), false);
        lo.fill(this->min);
        hi.fill(this->max);

        const Tensor mask = ((*this->tx) > lo) * ((*this->tx) < hi);
        const Tensor grad_expanded = _grad * mask;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

transpose::transpose(const GradientPacked &_x) : GradientFlow("Transpose") {
    this->tx = new Tensor(_x);
}

transpose::~transpose() {
    delete this->tx;
}

void transpose::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded = _grad.transpose();

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

slice::slice(const GradientPacked &_x, const i64 dim, const i64 start, const i64 end) : GradientFlow("Slice") {
    this->tx = new Tensor(_x);
    this->dim = dim;
    this->start = start;
    this->end = end;
}

slice::~slice() {
    delete this->tx;
}

void slice::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor grad_expanded(this->tx->shape(), this->tx->device(), false);
        grad_expanded.zero();

        //grad_expanded.slice_assign(this->dim, this->start, this->end, _grad);

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

mean::mean(const GradientPacked &_x) : GradientFlow("Mean") {
    this->tx = new Tensor(_x);
}

mean::~mean() {
    delete this->tx;
}

void mean::backward(const Tensor &_grad) {
    if (this->tx->is_require()) [[likely]] {
        const Tensor ones(this->tx->shape(), this->tx->device());
        ones.ones();

        const f32 N = static_cast<f32>(this->tx->len());
        const Tensor grad_expanded = _grad * (ones / N);

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

sum_dim::sum_dim(const GradientPacked &_x, const std::initializer_list<i64> dims, const bool keep_dim) : GradientFlow("SumDim") {
    this->tx = new Tensor(_x);
    this->dims = dims;
    this->keep_dim = keep_dim;
}

sum_dim::~sum_dim() {
    delete this->tx;
}

void sum_dim::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        Tensor grad_reshaped = _grad;

        if (!this->keep_dim) {
            for (const auto item : this->dims) {
                grad_reshaped = grad_reshaped.unsqueeze(item);
            }
        }

        const Tensor ones(this->tx->shape(), this->tx->device(), false);
        ones.ones();

        const Tensor grad_expanded = grad_reshaped * ones;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

mean_dim::mean_dim(const GradientPacked &_x, const std::initializer_list<i64> dims, const bool keep_dim): GradientFlow("MeanDim") {
    this->tx = new Tensor(_x);
    this->dims = dims;
    this->keep_dim = keep_dim;
}

mean_dim::~mean_dim() {
    delete this->tx;
}

void mean_dim::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        Tensor grad_reshaped = _grad;
        if (!this->keep_dim) {
            for (const auto item : this->dims) {
                grad_reshaped = grad_reshaped.unsqueeze(item);
            }
        }

        i64 total_elements = 1;
        for (const auto item : this->dims) {
            total_elements *= this->tx->shape()[item];
        }

        const Tensor ones(this->tx->shape(), this->tx->device(), false);
        ones.ones();

        const Tensor grad_expanded = grad_reshaped * (ones / static_cast<f32>(total_elements));
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

variance_dim::variance_dim(const GradientPacked &_x, const std::initializer_list<i64> dims, const bool keep_dim) : GradientFlow("VarianceDim") {
    this->tx = new Tensor(_x);
    this->dims = dims;
    this->keep_dim = keep_dim;
}

variance_dim::~variance_dim() {
    delete this->tx;
}

void variance_dim::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor mean_x = this->tx->mean(this->dims, true);
        const Tensor diff = (*this->tx) - mean_x;

        i64 N = 1;
        for (const auto item : this->dims) {
            N *= this->tx->shape()[item];
        }

        const Tensor grad_factor = diff * (2.0f / static_cast<f32>(N));

        Tensor grad_reshaped = _grad;
        if (!this->keep_dim) {
            for (const auto item : this->dims) {
                grad_reshaped = grad_reshaped.unsqueeze(item);
            }
        }

        const Tensor grad_expanded = grad_reshaped * grad_factor;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

stdv_dim::stdv_dim(const GradientPacked &_x, const GradientPacked& output, const std::initializer_list<i64> dims, const bool keep_dim) : GradientFlow("StdvDim") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(output);
    this->dims = dims;
    this->keep_dim = keep_dim;
}

stdv_dim::~stdv_dim() {
    delete this->tx;
    delete this->cached_output;
}

void stdv_dim::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor mean_x = this->tx->mean(this->dims, true);
        const Tensor diff = (*this->tx) - mean_x;

        i64 N = 1;
        for (const auto item : this->dims) {
            N *= this->tx->shape()[item];
        }

        const Tensor grad_factor = diff / ((*this->cached_output) * static_cast<f32>(N));

        Tensor grad_reshaped = _grad;
        if (!this->keep_dim) {
            for (const auto item : this->dims) {
                grad_reshaped = grad_reshaped.unsqueeze(item);
            }
        }

        const Tensor grad_expanded = grad_reshaped * grad_factor;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

norm1_dim::norm1_dim(const GradientPacked &_x, const std::initializer_list<i64> dims, const bool keep_dim) : GradientFlow("Norm1D") {
    this->tx = new Tensor(_x);
    this->dims = dims;
    this->keep_dim = keep_dim;
}

norm1_dim::~norm1_dim() {
    delete this->tx;
}

void norm1_dim::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor grad_factor = this->tx->sign();

        Tensor grad_reshaped = _grad;
        if (!this->keep_dim) {
            for (const auto item : this->dims) {
                grad_reshaped = grad_reshaped.unsqueeze(item);
            }
        }

        const Tensor grad_expanded = grad_reshaped * grad_factor;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

norm2_dim::norm2_dim(const GradientPacked &_x, const GradientPacked& output, const std::initializer_list<i64> dims, const bool keep_dim) : GradientFlow("Norm2D") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(output);
    this->dims = dims;
    this->keep_dim = keep_dim;
}

norm2_dim::~norm2_dim() {
    delete this->tx;
    delete this->cached_output;
}

void norm2_dim::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {

        const Tensor grad_factor = (*this->tx) / (*this->cached_output);

        Tensor grad_reshaped = _grad;
        if (!this->keep_dim) {
            for (const auto d : this->dims) {
                grad_reshaped = grad_reshaped.unsqueeze(d);
            }
        }

        const Tensor grad_expanded = grad_reshaped * grad_factor;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

variance::variance(const GradientPacked &_x) : GradientFlow("Variance") {
    this->tx = new Tensor(_x);
}

variance::~variance() {
    delete this->tx;
}

void variance::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const f32 N = static_cast<f32>(this->tx->len());
        const Tensor mean_x = this->tx->mean();

        const Tensor diff = (*this->tx) - mean_x;

        const Tensor grad_factor = diff * (2.0f / N);

        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

stdv::stdv(const GradientPacked &_x, const GradientPacked &output) : GradientFlow("Stdv") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(output);
}

stdv::~stdv() {
    delete this->tx;
    delete this->cached_output;
}

void stdv::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const f32 N = static_cast<f32>(this->tx->len());
        const Tensor mean_x = this->tx->mean();
        const Tensor diff = (*this->tx) - mean_x;

        const Tensor grad_factor = diff / (*this->cached_output * N);

        const Tensor grad_expanded = _grad * grad_factor;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

norm1::norm1(const GradientPacked &_x) : GradientFlow("Norm1") {
    this->tx = new Tensor(_x);
}

norm1::~norm1() {
    delete this->tx;
}

void norm1::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor grad_expanded = _grad * this->tx->sign();
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

norm2::norm2(const GradientPacked &_x, const GradientPacked &output) : GradientFlow("Norm2") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(output);
}

norm2::~norm2() {
    delete this->tx;
    delete this->cached_output;
}

void norm2::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor grad_factor = (*this->tx) / (*this->cached_output);

        const Tensor grad_expanded = _grad * grad_factor;
        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

exp2::exp2(const GradientPacked &_x) : GradientFlow("Exp2") {
    this->tx = new Tensor(_x);
}

exp2::~exp2() {
    delete this->tx;
}

void exp2::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        constexpr f32 ln2 = 0.69314718f;

        const Tensor grad_factor = this->tx->exp10() * ln2;
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

exp10::exp10(const GradientPacked &_x) : GradientFlow("Exp10") {
    this->tx = new Tensor(_x);
}

exp10::~exp10() {
    delete this->tx;
}

void exp10::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        constexpr f32 ln10 = 2.30258509f;

        const Tensor grad_factor = this->tx->exp10() * ln10;
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

log2::log2(const GradientPacked &_x) : GradientFlow("Log2") {
    this->tx = new Tensor(_x);
}

log2::~log2() {
    delete this->tx;
}

void log2::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        constexpr f32 ln2 = 0.69314718f;
        const Tensor grad_factor = (*this->tx * ln2).inv();
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

log10::log10(const GradientPacked &_x) : GradientFlow("Log10") {
    this->tx = new Tensor(_x);
}

log10::~log10() {
    delete this->tx;
}

void log10::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        constexpr f32 ln10 = 2.30258509f;
        const Tensor grad_factor = (*this->tx * ln10).inv();
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

square::square(const GradientPacked &_x) : GradientFlow("Square") {
    this->tx = new Tensor(_x);
}

square::~square() {
    delete this->tx;
}

void square::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor grad_factor = (*this->tx) * 2.0f;
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

tan::tan(const GradientPacked &_x) : GradientFlow("Tan") {
    this->tx = new Tensor(_x);
}

tan::~tan() {
    delete this->tx;
}

void tan::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor cos_x = this->tx->cos();
        const Tensor grad_factor = cos_x.square().inv();
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

cot::cot(const GradientPacked &_x) : GradientFlow("Cot") {
    this->tx = new Tensor(_x);
}

cot::~cot() {
    delete this->tx;
}

void cot::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor sin_x = this->tx->sin();
        const Tensor grad_factor = sin_x.square().inv().neg();
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

sign::sign(const GradientPacked &_x) : GradientFlow("Sign") {
    this->tx = new Tensor(_x);
}

sign::~sign() {
    delete this->tx;
}

void sign::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        const Tensor grad_expanded(this->tx->shape(), this->tx->device(), false);
        grad_expanded.zero();

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

erf::erf(const GradientPacked &_x) : GradientFlow("Erf") {
    this->tx = new Tensor(_x);
}

erf::~erf() {
    delete this->tx;
}

void erf::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {
        constexpr f32 factor = 1.12837916f;

        const Tensor exp_neg_x2 = (this->tx->square().neg()).exp();
        const Tensor grad_factor = exp_neg_x2 * factor;
        const Tensor grad_expanded = _grad * grad_factor;

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

inv::inv(const GradientPacked &_x) : GradientFlow("Inv") {
    this->tx = new Tensor(_x);
}

inv::~inv() {
    delete this->tx;
}

void inv::backward(const Tensor &_grad) {
    if (this->tx->is_require()) {

        const Tensor grad_factor = this->tx->square().inv().neg();
        const Tensor grad_expanded = _grad * grad_factor;

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
        const Tensor zeros(this->tx->shape(), this->tx->device(), false);
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
        const Tensor ones(this->cached_output->shape(), this->cached_output->device());
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
        const Tensor ones(this->cached_output->shape(), this->cached_output->device());
        ones.ones();

        const Tensor one_minus_output = ones - (*this->cached_output);
        const Tensor grad_coeff = (*this->cached_output) * one_minus_output;
        const Tensor grad_expanded = _grad * grad_coeff;

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

        const Tensor zeros(this->tx->shape(), this->tx->device());
        zeros.zero();

        const Tensor mask = (*this->tx) > zeros;

        const Tensor grad_coeff = mask * 1.0f + (1.0f - mask) * this->alpha;

        const Tensor grad_expanded = _grad * grad_coeff;

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

        const Tensor x_sq = this->tx->pow();
        const Tensor pdf = (-0.5f * x_sq).exp() * INV_SQRT_2PI;

        const Tensor grad_coeff = (*this->cached_output) + (*this->tx) * pdf;

        const Tensor grad_expanded = _grad * grad_coeff;

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
        const Tensor ones(this->tx->shape(), this->tx->device());
        ones.ones();

        const Tensor one_minus_sigmoid = ones - (*this->cached_output);

        const Tensor x_times_term = (*this->tx) * one_minus_sigmoid;

        Tensor grad_coeff = ones + x_times_term;

        grad_coeff = (*this->cached_output) * grad_coeff;

        const Tensor grad_expanded = _grad * grad_coeff;

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
            .matmul(col.transpose());
            //.reshape(tw->shape());
        tw->grad() += grad_w;
    }

    if (tb->has_grad()) {
        const Tensor grad_b = d_out.sum({1}, false);
        tb->grad() += grad_b;
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

        const Tensor grad_expanded = (*this->cached_output) * (_grad - sum_term);

        this->tx->grad() += grad_expanded;
        this->tx->backward(grad_expanded);
    }
}

logit_loss::logit_loss(const GradientPacked &_x, const GradientPacked &_y) : GradientFlow("LogitLoss") {
    this->tx = new Tensor(_x);
    this->cached_output = new Tensor(_y);
}

logit_loss::~logit_loss() {
    delete this->tx;
    delete this->cached_output;
}

void logit_loss::backward(const Tensor &_grad) {
    if (this->tx->has_grad()) {

        Tensor grad_input(this->tx->shape(), this->tx->device(), false);

        const f32* pred_ptr = this->tx->get();
        const f32* target_ptr = this->cached_output->get();
        f32* grad_in_ptr = grad_input.get();

        const size_t batch_size = this->tx->shape()[0];
        const size_t class_count = this->tx->shape()[1];
        const f32 inv_b = 1.0f / static_cast<f32>(batch_size);


        for (size_t b = 0; b < batch_size; ++b) {
            const size_t offset = b * class_count;

            f32 max_logit = pred_ptr[offset];
            for (size_t c = 1; c < class_count; ++c) {
                if (pred_ptr[offset + c] > max_logit) max_logit = pred_ptr[offset + c];
            }

            f32 sum_exp = 0.0f;
            for (size_t c = 0; c < class_count; ++c) {
                sum_exp += std::exp(pred_ptr[offset + c] - max_logit);
            }

            const f32 upstream_grad = _grad.empty() ? 1.0f : _grad.get()[0];

            for (size_t c = 0; c < class_count; ++c) {
                const f32 softmax_val = std::exp(pred_ptr[offset + c] - max_logit) / sum_exp;
                grad_in_ptr[offset + c] = (softmax_val - target_ptr[offset + c]) * upstream_grad * inv_b;
            }
        }

        this->tx->grad() += grad_input;
        this->tx->backward(grad_input);
    }
}